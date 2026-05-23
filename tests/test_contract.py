import operator
import base64
import inspect
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from typing import get_type_hints


class GraphContractTests(unittest.TestCase):
    def test_history_uses_append_reducer(self):
        from state import ThumbnailState

        hints = get_type_hints(ThumbnailState, include_extras=True)
        history_hint = hints["history"]

        self.assertIn(operator.add, history_hint.__metadata__)
        self.assertIn("design_strategy", hints)
        self.assertIn("route", hints)

    def test_should_continue_routes_on_rating_or_iteration_cap(self):
        from nodes import should_continue

        self.assertEqual(
            should_continue(
                {
                    "rating": 8,
                    "target_rating": 8,
                    "iteration": 1,
                    "max_iterations": 3,
                }
            ),
            "saver",
        )
        self.assertEqual(
            should_continue(
                {
                    "rating": 4,
                    "target_rating": 8,
                    "iteration": 3,
                    "max_iterations": 3,
                }
            ),
            "saver",
        )
        self.assertEqual(
            should_continue(
                {
                    "rating": 6,
                    "target_rating": 8,
                    "iteration": 2,
                    "max_iterations": 3,
                }
            ),
            "prompt_writer",
        )

    def test_build_graph_compiles_required_nodes_and_conditional_edge(self):
        import inspect
        import graph as graph_module

        compiled = graph_module.build_graph()
        drawable = compiled.get_graph()

        expected_nodes = [
            "web_search",
            "design_strategy",
            "prompt_writer",
            "generator",
            "critic",
            "should_continue",
            "saver",
        ]
        for node_name in expected_nodes:
            self.assertIn(node_name, drawable.nodes)

        source = inspect.getsource(graph_module.build_graph)
        self.assertGreaterEqual(source.count(".add_node("), 7)
        self.assertIn('"web_search", "design_strategy"', source)
        self.assertIn('"design_strategy", "prompt_writer"', source)
        self.assertIn('"critic", "should_continue"', source)
        self.assertIn('"should_continue",', source)
        self.assertIn(".add_conditional_edges(", source)
        self.assertIn(".compile()", source)

    def test_saver_writes_best_image_and_full_report(self):
        from nodes import node_saver

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            first = run_dir / "iter_1.png"
            second = run_dir / "iter_2.png"
            first.write_bytes(b"first-image")
            second.write_bytes(b"best-image")

            result = node_saver(
                {
                    "topic": "Why Python is the best language for AI",
                    "run_dir": str(run_dir),
                    "search_summary": "- Python owns the AI ecosystem.",
                    "target_rating": 8,
                    "iteration": 2,
                    "history": [
                        {
                            "iteration": 1,
                            "prompt": "first prompt",
                            "image_path": str(first),
                            "rating": 5,
                            "critique": "Too generic.",
                        },
                        {
                            "iteration": 2,
                            "prompt": "second prompt",
                            "image_path": str(second),
                            "rating": 7,
                            "critique": "More concrete.",
                        },
                    ],
                }
            )

            final_path = run_dir / "final.png"
            report_path = run_dir / "report.md"

            self.assertEqual(final_path.read_bytes(), b"best-image")
            self.assertEqual(result["final_image"], str(final_path))
            self.assertEqual(result["final_report"], str(report_path))

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Total iterations: 2", report)
            self.assertIn("Iteration 1", report)
            self.assertIn("Iteration 2", report)
            self.assertIn("Best rating: 7/10", report)

    def test_generator_handles_dalle_payload_without_response_format(self):
        import nodes

        source = inspect.getsource(nodes.node_generator)
        self.assertIn('"dall-e-3"', source)
        self.assertNotIn("response_format=", source)
        self.assertNotIn("style=", source)

        data = SimpleNamespace(b64_json=base64.b64encode(b"png-bytes").decode("ascii"), url=None)
        self.assertEqual(nodes._image_bytes_from_generation(data), b"png-bytes")


if __name__ == "__main__":
    unittest.main()
