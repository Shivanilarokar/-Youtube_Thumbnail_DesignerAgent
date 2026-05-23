"""Compiled LangGraph topology for the thumbnail reflexion agent."""

from langgraph.graph import END, START, StateGraph

from nodes import (
    collect_thumbnail_research,
    generate_thumbnail_image,
    record_loop_decision,
    review_thumbnail_image,
    save_best_thumbnail,
    should_continue,
    write_image_prompt,
)
from state import ThumbnailState


def build_graph():
    """Build and compile the required LangGraph state machine.

    Flow:
        START -> web_search -> prompt_writer -> generator -> critic
        -> should_continue -> prompt_writer or saver -> END
    """
    graph = StateGraph(ThumbnailState)

    # Strategy is intentionally omitted; should_continue is a visible router node.
    graph.add_node("web_search", collect_thumbnail_research)
    graph.add_node("prompt_writer", write_image_prompt)
    graph.add_node("generator", generate_thumbnail_image)
    graph.add_node("critic", review_thumbnail_image)
    graph.add_node("should_continue", record_loop_decision)
    graph.add_node("saver", save_best_thumbnail)

    graph.add_edge(START, "web_search")
    graph.add_edge("web_search", "prompt_writer")
    graph.add_edge("prompt_writer", "generator")
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", "should_continue")

    # The loop remains inside the compiled graph through this conditional edge.
    graph.add_conditional_edges(
        "should_continue",
        should_continue,
        {
            "prompt_writer": "prompt_writer",
            "saver": "saver",
        },
    )
    graph.add_edge("saver", END)

    return graph.compile()
