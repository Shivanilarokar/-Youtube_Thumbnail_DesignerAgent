from pathlib import Path

from graph import build_graph


def main() -> None:
    graph = build_graph()

    mermaid = graph.get_graph().draw_mermaid()
    Path("graph.mmd").write_text(mermaid, encoding="utf-8")

    png = graph.get_graph().draw_mermaid_png()
    Path("graph.png").write_bytes(png)
    print("Wrote graph.png")


if __name__ == "__main__":
    main()
