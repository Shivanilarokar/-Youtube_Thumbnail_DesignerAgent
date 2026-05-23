"""Generate graph.mmd and graph.png for the compiled LangGraph workflow."""

from pathlib import Path

from graph import build_graph


def main() -> None:
    """Render both Mermaid text and PNG diagram from the compiled graph."""
    graph = build_graph()

    # Writing the Mermaid file makes it easy to inspect the exact node names
    # when grading or debugging diagram rendering.
    mermaid = graph.get_graph().draw_mermaid()
    Path("graph.mmd").write_text(mermaid, encoding="utf-8")

    png = graph.get_graph().draw_mermaid_png()
    Path("graph.png").write_bytes(png)
    print("Wrote graph.mmd and graph.png")


if __name__ == "__main__":
    main()
