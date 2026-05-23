import os

from dotenv import load_dotenv
from tavily import TavilyClient


def web_search(query: str, max_results: int = 5) -> str:
    """Run one Tavily search and summarize hooks plus visual references."""
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to .env.")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_images=True,
        include_image_descriptions=True,
    )

    lines: list[str] = []
    for index, result in enumerate(response.get("results", []), start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "").strip()
        lines.append(f"{index}. {title} ({url})\n   {content[:300]}")

    images = response.get("images") or []
    if images:
        lines.append("\nVisual references:")
        for image in images[:5]:
            if isinstance(image, dict):
                description = image.get("description") or image.get("alt") or "No description"
                url = image.get("url", "")
                lines.append(f"- {description}: {url}")
            else:
                lines.append(f"- {image}")

    return "\n".join(lines) if lines else "No Tavily results returned."
