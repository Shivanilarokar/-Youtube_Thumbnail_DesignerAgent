"""LangGraph node functions for the YouTube thumbnail reflexion agent.

The graph owns the complete improvement loop. Each node accepts
`ThumbnailState`, returns a small dictionary of updates, and leaves routing to
the compiled LangGraph state machine in `graph.py`.
"""

import base64
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

from prompts import (
    CRITIC_SYSTEM,
    CRITIC_USER_TEXT,
    REVISION_HINT,
    ThumbnailPromptWriterSystem,
)
from state import ThumbnailState
from tools import web_search


OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
logger = logging.getLogger(__name__)


class ThumbnailReview(BaseModel):
    """Structured GPT-4o vision review for one generated thumbnail."""

    rating: int = Field(ge=1, le=10, description="Strict thumbnail quality score from 1 to 10.")
    critique: str = Field(description="Specific changes that should improve the next iteration.")


def _clean_slug(value: str, max_length: int = 60) -> str:
    """Convert a video topic into a safe folder-name fragment."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return (slug[:max_length] or "thumbnail").strip("_")


def _creative_text_model() -> ChatOpenAI:
    """Create the chat model used for image-prompt writing."""
    load_dotenv()
    return ChatOpenAI(model="gpt-4o", temperature=0.65)


def _vision_review_model():
    """Create GPT-4o vision with guaranteed Pydantic structured output."""
    load_dotenv()
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    return model.with_structured_output(ThumbnailReview, method="json_schema")


def _clean_image_prompt(raw_prompt: str) -> str:
    """Remove common LLM labels so only the image prompt reaches GPT Image 1."""
    prompt = raw_prompt.strip()
    return re.sub(
        r"^\s*\*{0,2}(?:thumbnail\s+)?prompt\*{0,2}\s*:\s*\*{0,2}\s*",
        "",
        prompt,
        flags=re.I,
    )


def _apply_strict_rating_cap(rating: int, critique: str) -> tuple[int, str]:
    """Keep the critic harsh when its own critique names a real fix."""
    fix_language = (
        "however",
        "but",
        "consider",
        "improve",
        "could",
        "should",
        "needs",
        "weak",
        "hard to read",
        "clutter",
    )
    has_fix = any(term in critique.lower() for term in fix_language)

    if rating > 7 and has_fix:
        note = (
            "\n\nStrict scoring cap applied: a thumbnail with a concrete "
            "readability, emotion, contrast, or composition fix cannot score above 7."
        )
        return 7, critique.rstrip() + note

    return rating, critique


def collect_thumbnail_research(state: ThumbnailState) -> dict:
    """Run the single Tavily research pull and initialize output state.

    This node runs once before the loop. It stores a compact research summary,
    creates the run folder, and resets iteration-related fields.
    """
    topic = state["topic"]
    logger.info("web_search: collecting hooks and visual references for %r", topic)

    summary = web_search(
        f"YouTube thumbnail hooks, title angles, visual metaphors, and visual references for: {topic}"
    )
    run_dir = OUTPUTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{_clean_slug(topic)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("web_search: output folder is %s", run_dir)
    return {
        "search_summary": summary,
        "run_dir": str(run_dir),
        "iteration": 0,
        "rating": 0,
        "critique": "",
        "routing_decision": "",
    }


def write_image_prompt(state: ThumbnailState) -> dict:
    """Write the next GPT Image 1 prompt.

    On iteration one, this uses the topic and Tavily research. On later visits,
    the latest GPT-4o critique is attached so the prompt changes materially.
    """
    logger.info("prompt_writer: drafting prompt for iteration %d", state.get("iteration", 0) + 1)

    critique = state.get("critique", "")
    feedback = (
        REVISION_HINT.format(
            rating=state.get("rating", 0),
            critique=critique,
        )
        if critique
        else ""
    )

    system_prompt = ThumbnailPromptWriterSystem.format(
        user_topic=state["topic"],
    )
    user_prompt = "\n\n".join(
        part
        for part in (
            f"Video topic:\n{state['topic']}",
            f"Research bullets:\n{state['search_summary']}",
            feedback,
            "Write the image prompt.",
        )
        if part
    )
    response = _creative_text_model().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    prompt = _clean_image_prompt(str(response.content))
    logger.info("prompt_writer: wrote %d characters", len(prompt))
    return {"current_prompt": prompt}


def _decode_image_bytes(image_data: Any) -> bytes:
    """Decode the base64 PNG returned by GPT Image 1."""
    image_b64 = getattr(image_data, "b64_json", None)
    if not image_b64:
        raise RuntimeError("OpenAI image generation returned no b64_json image data.")
    return base64.b64decode(image_b64)


def generate_thumbnail_image(state: ThumbnailState) -> dict:
    """Generate one thumbnail with GPT Image 1 and write `iter_N.png`.

    This node is inside the reflexion loop, so every visit increments
    `iteration` and creates a new image artifact.
    """
    load_dotenv()
    next_iteration = state.get("iteration", 0) + 1
    out_path = Path(state["run_dir"]) / f"iter_{next_iteration}.png"

    logger.info("generator: creating %s with gpt-image-1", out_path.name)
    response = OpenAI().images.generate(
        model="gpt-image-1",
        prompt=state["current_prompt"],
        size="1536x1024",
        quality="medium",
        n=1,
    )

    out_path.write_bytes(_decode_image_bytes(response.data[0]))
    logger.info("generator: saved %s", out_path)
    return {"image_path": str(out_path), "iteration": next_iteration}


def review_thumbnail_image(state: ThumbnailState) -> dict:
    """Critique the latest PNG with GPT-4o vision and append review history.

    The image is passed as a base64 data URL inside a `HumanMessage`. LangChain
    parses the response through `ThumbnailReview`, so `rating` is always an int.
    """
    image_path = Path(state["image_path"])
    logger.info("critic: reviewing %s", image_path.name)

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:image/png;base64,{image_b64}"

    previous_critique = state.get("critique", "")
    critic_system = CRITIC_SYSTEM.format(
        topic=state["topic"],
        prompt=state["current_prompt"],
        previous_critique=previous_critique or "None",
    )

    review: ThumbnailReview = _vision_review_model().invoke(
        [
            SystemMessage(content=critic_system),
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

    capped_rating, capped_critique = _apply_strict_rating_cap(
        int(review.rating),
        review.critique,
    )

    record = {
        "iteration": state["iteration"],
        "prompt": state["current_prompt"],
        "image_path": state["image_path"],
        "rating": capped_rating,
        "critique": capped_critique,
    }

    logger.info("critic: iteration %d scored %d/10", state["iteration"], capped_rating)
    return {
        "rating": capped_rating,
        "critique": capped_critique,
        "history": [record],
    }


def _next_graph_step(state: ThumbnailState) -> str:
    """Return the next node name based on score and iteration limits."""
    rating = int(state.get("rating", 0))
    iteration = int(state.get("iteration", 0))
    target_rating = int(state.get("target_rating", 8))
    max_iterations = int(state.get("max_iterations", 3))
    min_iterations = min(int(state.get("min_iterations", 3)), max_iterations)

    if iteration < min_iterations:
        return "prompt_writer"
    if rating >= target_rating or iteration >= max_iterations:
        return "saver"
    return "prompt_writer"


def record_loop_decision(state: ThumbnailState) -> dict:
    """Graph node that records the loop route for diagram clarity."""
    decision = _next_graph_step(state)
    logger.info(
        "should_continue: rating=%s target=%s iteration=%s/%s next=%s",
        state.get("rating", 0),
        state.get("target_rating", 8),
        state.get("iteration", 0),
        state.get("max_iterations", 3),
        decision,
    )
    return {"routing_decision": decision}


def should_continue(state: ThumbnailState) -> str:
    """Conditional-edge function used by `add_conditional_edges`.

    LangGraph calls this after the visible `should_continue` node.
    """
    return state.get("routing_decision") or _next_graph_step(state)


def save_best_thumbnail(state: ThumbnailState) -> dict:
    """Copy the best iteration to `final.png` and write `report.md`."""
    history = state.get("history", [])
    if not history:
        raise RuntimeError("Cannot save final output because history is empty.")

    run_dir = Path(state["run_dir"])
    best = max(history, key=lambda item: (int(item["rating"]), int(item["iteration"])))
    final_path = run_dir / "final.png"
    report_path = run_dir / "report.md"

    logger.info(
        "saver: selected iteration %s with rating %s/10",
        best["iteration"],
        best["rating"],
    )
    shutil.copyfile(best["image_path"], final_path)

    lines = [
        f"# YouTube Thumbnail Reflexion Report: {state['topic']}",
        "",
        f"Best rating: {best['rating']}/10",
        f"Best iteration: {best['iteration']}",
        f"Total iterations: {state['iteration']}",
        f"Target rating: {state.get('target_rating', 8)}/10",
        f"Minimum iterations before save: {state.get('min_iterations', 3)}",
        f"Max iterations: {state.get('max_iterations', 3)}",
        "",
        "## Search Summary",
        "",
        state.get("search_summary", "No search summary."),
    ]

    lines.extend(["", "## Iterations", ""])

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
    logger.info("saver: wrote %s and %s", final_path.name, report_path.name)
    return {"final_image": str(final_path), "final_report": str(report_path)}
