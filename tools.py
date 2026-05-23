"""Tavily search integration for the thumbnail reflexion graph.

This module intentionally keeps the web-search surface small: the graph needs
one research pull, formatted as plain text, so downstream LLM nodes can use it
without depending on Tavily response internals.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient


logger = logging.getLogger(__name__)


def _get_tavily_api_key() -> str:
    """Load and return the Tavily API key from `.env` or the process env.

    Raises:
        RuntimeError: If `TAVILY_API_KEY` is missing. The message tells the user
            exactly how to fix local setup without exposing any secret value.
    """
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY is missing; Tavily search cannot run.")
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to .env.")

    logger.debug("Tavily API key loaded from environment.")
    return api_key


def _format_search_response(response: dict[str, Any], max_images: int = 5) -> str:
    """Convert a Tavily response dict into LLM-readable research notes.

    Args:
        response: Raw response returned by `TavilyClient.search(...)`.
        max_images: Maximum number of image references to include.

    Returns:
        A newline-separated summary containing numbered search results and, when
        available, visual references for the thumbnail strategy node.
    """
    lines: list[str] = []

    # Search results provide hooks, claims, and source URLs for the prompt
    # strategy. Keep snippets short so the graph state remains compact.
    for index, result in enumerate(response.get("results", []), start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "").strip()
        lines.append(f"{index}. {title} ({url})\n   {content[:300]}")

    # Tavily can return either image dictionaries or plain image URLs depending
    # on API response shape. Preserve both formats for visual inspiration.
    images = response.get("images") or []
    if images:
        lines.append("\nVisual references:")
        for image in images[:max_images]:
            if isinstance(image, dict):
                description = image.get("description") or image.get("alt") or "No description"
                url = image.get("url", "")
                lines.append(f"- {description}: {url}")
            else:
                lines.append(f"- {image}")

    if not lines:
        logger.warning("Tavily returned no search results or image references.")
        return "No Tavily results returned."

    logger.debug(
        "Formatted Tavily response with %d text results and %d image references.",
        len(response.get("results", [])),
        len(images),
    )
    return "\n".join(lines)


def web_search(query: str, max_results: int = 5) -> str:
    """Run the one Tavily search used by the LangGraph agent.

    The `web_search` node calls this once at the start of a run. It pulls hooks,
    topic context, source URLs, and image references, then returns a compact text
    summary stored in `state["search_summary"]`.

    Args:
        query: Search query generated from the user's video topic.
        max_results: Number of Tavily text results to request.

    Returns:
        A plain-text research summary for downstream LLM nodes.

    Raises:
        RuntimeError: If the API key is missing or the Tavily request fails.
    """
    api_key = _get_tavily_api_key()
    logger.info("Running Tavily search for query=%r max_results=%d", query, max_results)

    client = TavilyClient(api_key=api_key)
    try:
        # `include_images` and `include_image_descriptions` give the design
        # strategy node visual references, not just text hooks.
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_images=True,
            include_image_descriptions=True,
        )
    except Exception as exc:
        logger.exception("Tavily search failed for query=%r", query)
        raise RuntimeError(f"Tavily search failed: {exc}") from exc

    summary = _format_search_response(response)
    logger.info("Tavily search formatted into %d characters.", len(summary))
    return summary


def _demo_search() -> None:
    """Run a small manual Tavily check when executing `python tools.py`."""
    test_query = "Best practices for YouTube thumbnail design"
    try:
        result = web_search(test_query)
        print("Tavily Search Summary:\n", result)
    except RuntimeError as exc:
        print("Error during Tavily search:", exc)


if __name__ == "__main__":
    _demo_search()
