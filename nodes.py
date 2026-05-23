import base64
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, Field

from prompts import (
    CRITIC_SYSTEM,
    CRITIC_USER_TEXT,
    DESIGN_STRATEGY_SYSTEM,
    DESIGN_STRATEGY_USER,
    PROMPT_WRITER_SYSTEM,
    PROMPT_WRITER_USER,
)
from state import ThumbnailState
from tools import web_search


OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


class CritiqueOutput(BaseModel):
    rating: int = Field(ge=1, le=10, description="Strict thumbnail quality score from 1 to 10.")
    critique: str = Field(description="Actionable critique explaining how to improve the next iteration.")


class DesignStrategyOutput(BaseModel):
    focal_subject: str = Field(description="The single dominant subject viewers notice first.")
    secondary_visual: str = Field(description="A supporting visual element that clarifies the topic.")
    thumbnail_text: str = Field(description="A 2-5 word text overlay.")
    text_position: str = Field(description="Exact text placement in the frame.")
    background: str = Field(description="Concrete background environment.")
    lighting: str = Field(description="Lighting direction, intensity, and color temperature.")
    color_contrast: str = Field(description="Specific color separation to make the image pop.")
    mood: str = Field(description="Viewer emotion the thumbnail should create.")
    curiosity_hook: str = Field(description="The visual reason someone would click.")


def _slugify(value: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return (slug[:max_length] or "thumbnail").strip("_")


def _writer_llm() -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.8)


def _critic_llm():
    load_dotenv()
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    return model.with_structured_output(CritiqueOutput, method="json_schema")


def _design_strategy_llm():
    return _writer_llm().with_structured_output(DesignStrategyOutput, method="json_schema")


def node_web_search(state: ThumbnailState) -> dict:
    topic = state["topic"]
    summary = web_search(
        f"YouTube thumbnail hooks, title angles, visual metaphors, and visual references for: {topic}"
    )

    run_dir = OUTPUTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{_slugify(topic)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return {
        "search_summary": summary,
        "run_dir": str(run_dir),
        "iteration": 0,
        "rating": 0,
        "critique": "",
    }


def node_design_strategy(state: ThumbnailState) -> dict:
    user_prompt = DESIGN_STRATEGY_USER.format(
        topic=state["topic"],
        search_summary=state["search_summary"],
    )
    design_strategy: DesignStrategyOutput = _design_strategy_llm().invoke(
        [
            SystemMessage(content=DESIGN_STRATEGY_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )
    return {"design_strategy": design_strategy.model_dump()}


def node_prompt_writer(state: ThumbnailState) -> dict:
    critique = state.get("critique", "")
    if critique:
        critique_block = f'Previous rating: {state.get("rating", 0)}/10\n"{critique}"'
    else:
        critique_block = "None. This is the first draft."

    user_prompt = PROMPT_WRITER_USER.format(
        topic=state["topic"],
        search_summary=state["search_summary"],
        design_strategy="\n".join(
            f"- {key}: {value}" for key, value in state["design_strategy"].items()
        ),
        critique_block=critique_block,
    )
    response = _writer_llm().invoke(
        [
            SystemMessage(content=PROMPT_WRITER_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )
    return {"current_prompt": str(response.content).strip()}


def _image_bytes_from_generation(image_data) -> bytes:
    image_b64 = getattr(image_data, "b64_json", None)
    if image_b64:
        return base64.b64decode(image_b64)

    image_url = getattr(image_data, "url", None)
    if image_url:
        with urlopen(image_url, timeout=60) as response:
            return response.read()

    raise RuntimeError("OpenAI image generation returned neither b64_json nor url.")


def node_generator(state: ThumbnailState) -> dict:
    load_dotenv()
    next_iteration = state.get("iteration", 0) + 1
    out_path = Path(state["run_dir"]) / f"iter_{next_iteration}.png"
    client = OpenAI()

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=state["current_prompt"],
            size="1792x1024",
            quality="standard",
            n=1,
        )
    except BadRequestError as exc:
        message = str(exc)
        if "dall-e-3" not in message or "does not exist" not in message:
            raise
        response = client.images.generate(
            model="gpt-image-1",
            prompt=state["current_prompt"],
            size="1536x1024",
            quality="medium",
            n=1,
        )

    out_path.write_bytes(_image_bytes_from_generation(response.data[0]))
    return {"image_path": str(out_path), "iteration": next_iteration}


def node_critic(state: ThumbnailState) -> dict:
    image_bytes = Path(state["image_path"]).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{image_b64}"

    result: CritiqueOutput = _critic_llm().invoke(
        [
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": CRITIC_USER_TEXT.format(
                            topic=state["topic"],
                            prompt=state["current_prompt"],
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            ),
        ]
    )

    record = {
        "iteration": state["iteration"],
        "prompt": state["current_prompt"],
        "image_path": state["image_path"],
        "rating": int(result.rating),
        "critique": result.critique,
    }

    return {
        "rating": int(result.rating),
        "critique": result.critique,
        "history": [record],
    }


def should_continue(state: ThumbnailState) -> str:
    rating = int(state.get("rating", 0))
    iteration = int(state.get("iteration", 0))
    target_rating = int(state.get("target_rating", 8))
    max_iterations = int(state.get("max_iterations", 3))

    if rating >= target_rating:
        return "saver"
    if iteration >= max_iterations:
        return "saver"
    return "prompt_writer"


def node_should_continue(state: ThumbnailState) -> dict:
    return {"route": should_continue(state)}


def node_saver(state: ThumbnailState) -> dict:
    history = state.get("history", [])
    if not history:
        raise RuntimeError("Cannot save final output because history is empty.")

    run_dir = Path(state["run_dir"])
    best = max(history, key=lambda item: int(item["rating"]))

    final_path = run_dir / "final.png"
    shutil.copyfile(best["image_path"], final_path)

    report_path = run_dir / "report.md"
    lines = [
        f"# YouTube Thumbnail Reflexion Report: {state['topic']}",
        "",
        f"Best rating: {best['rating']}/10",
        f"Best iteration: {best['iteration']}",
        f"Total iterations: {state['iteration']}",
        f"Target rating: {state.get('target_rating', 8)}/10",
        f"Max iterations: {state.get('max_iterations', 3)}",
        "",
        "## Search Summary",
        "",
        state.get("search_summary", "No search summary."),
        "",
        "## Design Strategy",
        "",
    ]

    for key, value in state.get("design_strategy", {}).items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

    lines.extend(
        [
            "",
            "## Iterations",
            "",
        ]
    )

    for item in history:
        image_name = Path(item["image_path"]).name
        lines.extend(
            [
                f"### Iteration {item['iteration']}",
                "",
                f"Rating: {item['rating']}/10",
                "",
                "Prompt:",
                "",
                item["prompt"],
                "",
                "Critique:",
                "",
                item["critique"],
                "",
                f"Image: `{image_name}`",
                "",
                f"![Iteration {item['iteration']}](./{image_name})",
                "",
            ]
        )

    lines.extend(
        [
            "## Final Image",
            "",
            f"Selected iteration: {best['iteration']}",
            "",
            "![Final thumbnail](./final.png)",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"final_image": str(final_path), "final_report": str(report_path)}
