import argparse
import logging
import sys

from dotenv import load_dotenv

from graph import build_graph


DEFAULT_TOPIC = "Why Python is the best language for AI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the YouTube thumbnail reflexion agent.")
    parser.add_argument("topic", nargs="?", default=DEFAULT_TOPIC, help="Video topic to design a thumbnail for.")
    parser.add_argument("--target-rating", type=int, default=8, help="Stop when critic rating reaches this value.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum prompt/generate/critic loops.")
    parser.add_argument("--stream", action="store_true", help="Print LangGraph node updates as they happen.")
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    graph = build_graph()
    initial_state = {
        "topic": args.topic,
        "target_rating": args.target_rating,
        "max_iterations": args.max_iterations,
        "iteration": 0,
        "rating": 0,
        "critique": "",
        "history": [],
    }

    try:
        if args.stream:
            final_state = None
            for update in graph.stream(initial_state, stream_mode="updates"):
                print(update)
                final_state = update
            if final_state is None:
                raise RuntimeError("Graph produced no updates.")
        else:
            final = graph.invoke(initial_state)
            print(f"Final image: {final['final_image']}")
            print(f"Report: {final['final_report']}")
            print(f"Best score observed: {final['rating']}/10")
            print(f"Iterations: {final['iteration']}")
    except Exception as exc:
        logging.exception("Agent failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
