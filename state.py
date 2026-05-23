import operator
from typing import Annotated, Any, TypedDict


class ThumbnailState(TypedDict, total=False):
    topic: str
    target_rating: int
    max_iterations: int

    search_summary: str
    run_dir: str
    design_strategy: dict[str, Any]

    current_prompt: str
    image_path: str
    rating: int
    critique: str
    iteration: int

    final_image: str
    final_report: str
    route: str

    history: Annotated[list[dict[str, Any]], operator.add]
