from langgraph.graph import END, START, StateGraph

from nodes import (
    node_critic,
    node_generator,
    node_prompt_writer,
    node_saver,
    node_design_strategy,
    node_should_continue,
    node_web_search,
    should_continue,
)
from state import ThumbnailState


def build_graph():
    graph = StateGraph(ThumbnailState)

    graph.add_node("web_search", node_web_search)
    graph.add_node("design_strategy", node_design_strategy)
    graph.add_node("prompt_writer", node_prompt_writer)
    graph.add_node("generator", node_generator)
    graph.add_node("critic", node_critic)
    graph.add_node("should_continue", node_should_continue)
    graph.add_node("saver", node_saver)

    graph.add_edge(START, "web_search")
    graph.add_edge("web_search", "design_strategy")
    graph.add_edge("design_strategy", "prompt_writer")
    graph.add_edge("prompt_writer", "generator")
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", "should_continue")
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
