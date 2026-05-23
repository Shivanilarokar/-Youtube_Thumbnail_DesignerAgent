"""Shared LangGraph state for the thumbnail reflexion agent.

Every node receives this dictionary-like state and returns only the fields it
changed. LangGraph merges those updates between nodes. The `history` field uses
an append reducer so each critic pass adds one iteration record instead of
overwriting previous records.
"""

import operator
from typing import Annotated, Any, TypedDict


class ThumbnailState(TypedDict, total=False):
    """State carried through the compiled thumbnail-design graph."""

    # User-controlled run settings.
    topic: str
    target_rating: int
    min_iterations: int
    max_iterations: int

    # Research created before the improvement loop starts.
    search_summary: str
    run_dir: str

    # Current loop iteration artifacts.
    current_prompt: str
    image_path: str
    rating: int
    critique: str
    iteration: int
    routing_decision: str

    # Final files written by the saver node.
    final_image: str
    final_report: str

    # Append-only review log. Required by the assignment rubric.
    history: Annotated[list[dict[str, Any]], operator.add]
