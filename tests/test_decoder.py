"""Tests for JSON prefix validation and function-call schema checks."""
import json
import os
import tempfile
import unittest

from src.decoder import (
    JSONConstrainedDecoder,
    _is_valid_json_prefix,
)
from src.models import FunctionDefinition
from src.vocab import Vocabulary


def make_decoder() -> JSONConstrainedDecoder:
    function = FunctionDefinition.model_validate({
        "name": "fn_example",
        "description": "Exercise decoder validation.",
        "parameters": {
            "count": {"type": "number"},
            "label": {"type": "string"},
        },
        "returns": {"type": "string"},
    })
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False) as vocab_file:
        json.dump({}, vocab_file)
        vocab_path = vocab_file.name
    try:
        return JSONConstrainedDecoder(Vocabulary(vocab_path), [function])
    finally:
        os.unlink(vocab_path)


class JsonPrefixTests(unittest.TestCase):
    def test_accepts_complete_nested_json(self) -> None:
        self.assertTrue(_is_valid_json_prefix(
            r'{"items":[1,true,null,{"text":"line\n"}]}'
        ))

    def test_accepts_incomplete_but_possible_prefixes(self) -> None:
        for prefix in ('{"items":', '{"items":[1,', '{"text":"unfinished'):
            with self.subTest(prefix=prefix):
                self.assertTrue(_is_valid_json_prefix(prefix))

    def test_rejects_invalid_json_and_invalid_prefixes(self) -> None:
        for text in ('{"value":,}', '{"value":01}', '{"value":[1,]}',
                     '"unescaped\nnewline"'):
            with self.subTest(text=text):
                self.assertFalse(_is_valid_json_prefix(text))


class CallSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = make_decoder()

    def test_normalizes_valid_call(self) -> None:
        self.assertEqual(
            self.decoder.normalize_call({
                "name": "fn_example",
                "parameters": {"count": 3, "label": "ok"},
            }),
            {
                "name": "fn_example",
                "parameters": {"count": 3, "label": "ok"},
            },
        )

    def test_accepts_empty_string_with_all_required_parameters(self) -> None:
        self.assertEqual(
            self.decoder.normalize_call({
                "name": "fn_example",
                "parameters": {"count": 0, "label": ""},
            }),
            {
                "name": "fn_example",
                "parameters": {"count": 0, "label": ""},
            },
        )

    def test_accepts_large_numbers_exactly(self) -> None:
        large_integer = 10 ** 60
        large_call = {
            "name": "fn_example",
            "parameters": {"count": large_integer, "label": "large"},
        }
        self.assertEqual(
            self.decoder.normalize_call(large_call),
            large_call,
        )

    def test_accepts_special_characters_in_string_parameter(self) -> None:
        label = 'quote: " slash: \\\\ newline:\n unicode: café'
        call = {
            "name": "fn_example",
            "parameters": {"count": 1, "label": label},
        }
        serialized = json.dumps(call, ensure_ascii=False)
        generated_prefix = serialized[:-1]

        self.assertTrue(self.decoder._is_valid_prefix("", generated_prefix))
        self.assertTrue(self.decoder._is_valid_prefix(generated_prefix, "}"))
        self.assertEqual(
            self.decoder.normalize_call(json.loads(serialized)),
            call,
        )

    def test_accepts_json_numeric_string_and_normalizes_it(self) -> None:
        result = self.decoder.normalize_call({
            "name": "fn_example",
            "parameters": {"count": "3.5", "label": "ok"},
        })
        self.assertEqual(
            result,
            {
                "name": "fn_example",
                "parameters": {"count": 3.5, "label": "ok"},
            },
        )

    def test_rejects_wrong_envelope_and_function(self) -> None:
        for call in (
            [],
            {"name": "unknown", "parameters": {"count": 1, "label": "x"}},
            {
                "name": "fn_example",
                "parameters": {"count": 1, "label": "x"},
                "extra": True,
            },
            {"name": "fn_example", "parameters": []},
        ):
            with self.subTest(call=call):
                self.assertIsNone(self.decoder.normalize_call(call))

    def test_rejects_missing_extra_and_mistyped_parameters(self) -> None:
        for parameters in (
            {"count": 1},
            {"count": 1, "label": "x", "extra": 2},
            {"count": True, "label": "x"},
            {"count": "not a number", "label": "x"},
            {"count": 1, "label": 9},
        ):
            with self.subTest(parameters=parameters):
                self.assertIsNone(self.decoder.normalize_call({
                    "name": "fn_example",
                    "parameters": parameters,
                }))

    def test_rejects_non_finite_numbers(self) -> None:
        self.assertIsNone(self.decoder.normalize_call({
            "name": "fn_example",
            "parameters": {"count": float("inf"), "label": "x"},
        }))
        self.assertIsNone(self.decoder.normalize_call({
            "name": "fn_example",
            "parameters": {"count": "1e999", "label": "x"},
        }))

    def test_valid_call_predicate_matches_normalization(self) -> None:
        valid_call = {
            "name": "fn_example",
            "parameters": {"count": 2, "label": "x"},
        }
        self.assertTrue(self.decoder.is_valid_call(valid_call))
        self.assertFalse(self.decoder.is_valid_call({"name": "unknown"}))

    def test_rejects_extra_top_level_fields_after_parameters(self) -> None:
        complete_call = (
            '{"name": "fn_example", "parameters": '
            '{"count": 1, "label": "x"}'
        )
        self.assertFalse(self.decoder._is_valid_prefix(
            complete_call, ', "extra": true}'
        ))

    def test_accepts_valid_call_prefix_and_completion(self) -> None:
        self.assertTrue(self.decoder._is_valid_prefix("", '{"name": "'))
        self.assertTrue(self.decoder._is_valid_prefix(
            '{"name": "fn_example", "parameters": '
            '{"count": 1, "label": "x"',
            "}",
        ))


if __name__ == "__main__":
    unittest.main()
