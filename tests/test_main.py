"""Tests for input handling, validation, and output errors."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import __main__ as app


FUNCTIONS = [{
    "name": "fn_echo",
    "description": "Return the provided text.",
    "parameters": {"text": {"type": "string"}},
    "returns": {"type": "string"},
}]
PROMPTS = [{"prompt": "Echo hello"}]


class LoadJsonFileTests(unittest.TestCase):
    def test_loads_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text('[{"prompt": "hello"}]', encoding="utf-8")
            self.assertEqual(
                app.load_json_file(str(path)),
                [{"prompt": "hello"}],
            )

    def test_missing_file_exits_with_clear_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                app.load_json_file("/missing/input.json")
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Error loading /missing/input.json", stderr.getvalue())

    def test_invalid_json_exits_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{broken", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    app.load_json_file(str(path))
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Error loading", stderr.getvalue())

    def test_non_array_root_exits_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text('{"prompt": "hello"}', encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    app.load_json_file(str(path))
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("JSON root must be an array", stderr.getvalue())


class MainTests(unittest.TestCase):
    def write_inputs(
        self,
        directory: str,
        prompts: object = PROMPTS,
    ) -> tuple[str, str]:
        functions_path = Path(directory) / "functions.json"
        prompts_path = Path(directory) / "prompts.json"
        functions_path.write_text(json.dumps(FUNCTIONS), encoding="utf-8")
        prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
        return str(functions_path), str(prompts_path)

    def test_malformed_record_exits_without_model_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            functions_path, prompts_path = self.write_inputs(
                directory, prompts=["not an object"])
            stderr = io.StringIO()
            with (
                patch.object(
                    app.sys,
                    "argv",
                    ["src", "-functions_definition", functions_path,
                     "-input", prompts_path],
                ),
                patch.object(app, "Small_LLM_Model") as model_factory,
                contextlib.redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as raised:
                    app.main()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Error validating input data", stderr.getvalue())
        model_factory.assert_not_called()

    def test_writes_output_to_filename_without_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            functions_path, prompts_path = self.write_inputs(directory)
            output_path = Path(directory) / "results.json"
            stdout = io.StringIO()
            previous_directory = Path.cwd()
            try:
                os.chdir(directory)
                with (
                    patch.object(
                        app.sys,
                        "argv",
                        ["src", "-functions_definition", functions_path,
                         "-input", prompts_path, "-output", "results.json"],
                    ),
                    patch.object(
                        app,
                        "Small_LLM_Model",
                        return_value=SimpleNamespace(
                            get_path_to_vocab_file=lambda: "unused.json"
                        ),
                    ),
                    patch.object(
                        app, "Vocabulary"
                    ) as vocabulary_factory,
                    patch.object(
                        app,
                        "generate_function_call",
                        return_value={
                            "name": "fn_echo",
                            "parameters": {"text": "hello"},
                        },
                    ),
                    contextlib.redirect_stdout(stdout),
                ):
                    vocabulary = vocabulary_factory.return_value
                    vocabulary.get_all_tokens.return_value = []
                    app.main()
            finally:
                os.chdir(previous_directory)

            results = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(results, [{
            "prompt": "Echo hello",
            "name": "fn_echo",
            "parameters": {"text": "hello"},
        }])
        self.assertIn("Success! Output written to", stdout.getvalue())

    def test_unwritable_output_exits_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            functions_path, prompts_path = self.write_inputs(directory)
            output_path = Path(directory) / "existing-directory"
            output_path.mkdir()
            stderr = io.StringIO()
            with (
                patch.object(
                    app.sys,
                    "argv",
                    ["src", "-functions_definition", functions_path,
                     "-input", prompts_path, "-output", str(output_path)],
                ),
                patch.object(
                    app,
                    "Small_LLM_Model",
                    return_value=SimpleNamespace(
                        get_path_to_vocab_file=lambda: "unused.json"
                    ),
                ),
                patch.object(app, "Vocabulary") as vocabulary_factory,
                patch.object(
                    app,
                    "generate_function_call",
                    return_value={
                        "name": "fn_echo",
                        "parameters": {"text": "hello"},
                    },
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                vocabulary = vocabulary_factory.return_value
                vocabulary.get_all_tokens.return_value = []
                with self.assertRaises(SystemExit) as raised:
                    app.main()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn(
            f"Error writing output file {output_path}",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
