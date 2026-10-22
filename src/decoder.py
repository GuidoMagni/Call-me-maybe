"""Constrained decoding logic for structured JSON generation."""
import json
import math
import numpy as np
import numpy.typing as npt
from typing import Any, List
from src.models import FunctionDefinition
from src.vocab import Vocabulary


def _is_valid_json_prefix(text: str) -> bool:
    """Return whether text is a prefix of a syntactically valid JSON value."""
    length = len(text)

    def skip_whitespace(position: int) -> int:
        while position < length and text[position] in " \t\n\r":
            position += 1
        return position

    def parse_string(position: int) -> tuple[int, bool] | None:
        position += 1
        while position < length:
            char = text[position]
            if char == '"':
                return position + 1, True
            if ord(char) < 0x20:
                return None
            if char == "\\":
                position += 1
                if position == length:
                    return length, False
                escape = text[position]
                if escape in '"\\/bfnrt':
                    position += 1
                    continue
                if escape != "u":
                    return None
                position += 1
                for _ in range(4):
                    if position == length:
                        return length, False
                    if text[position] not in "0123456789abcdefABCDEF":
                        return None
                    position += 1
                continue
            position += 1
        return length, False

    def parse_number(position: int) -> tuple[int, bool] | None:
        if text[position] == "-":
            position += 1
            if position == length:
                return length, False
        if position == length:
            return length, False
        if text[position] == "0":
            position += 1
        elif "1" <= text[position] <= "9":
            position += 1
            while position < length and "0" <= text[position] <= "9":
                position += 1
        else:
            return None

        if position < length and text[position] == ".":
            position += 1
            if position == length:
                return length, False
            if not "0" <= text[position] <= "9":
                return None
            while position < length and "0" <= text[position] <= "9":
                position += 1

        if position < length and text[position] in "eE":
            position += 1
            if position < length and text[position] in "+-":
                position += 1
            if position == length:
                return length, False
            if not "0" <= text[position] <= "9":
                return None
            while position < length and "0" <= text[position] <= "9":
                position += 1
        return position, True

    def parse_value(position: int) -> tuple[int, bool] | None:
        position = skip_whitespace(position)
        if position == length:
            return length, False

        char = text[position]
        if char == '"':
            return parse_string(position)
        if char == "{":
            position = skip_whitespace(position + 1)
            if position == length:
                return length, False
            if text[position] == "}":
                return position + 1, True
            while True:
                if position == length:
                    return length, False
                if text[position] != '"':
                    return None
                key = parse_string(position)
                if key is None:
                    return None
                position, complete = key
                if not complete:
                    return length, False
                position = skip_whitespace(position)
                if position == length:
                    return length, False
                if text[position] != ":":
                    return None
                value = parse_value(position + 1)
                if value is None:
                    return None
                position, complete = value
                if not complete:
                    return length, False
                position = skip_whitespace(position)
                if position == length:
                    return length, False
                if text[position] == "}":
                    return position + 1, True
                if text[position] != ",":
                    return None
                position = skip_whitespace(position + 1)
        if char == "[":
            position = skip_whitespace(position + 1)
            if position == length:
                return length, False
            if text[position] == "]":
                return position + 1, True
            while True:
                value = parse_value(position)
                if value is None:
                    return None
                position, complete = value
                if not complete:
                    return length, False
                position = skip_whitespace(position)
                if position == length:
                    return length, False
                if text[position] == "]":
                    return position + 1, True
                if text[position] != ",":
                    return None
                position = skip_whitespace(position + 1)
        if char in "tfn":
            literal = {"t": "true", "f": "false", "n": "null"}[char]
            remaining = text[position:]
            if len(remaining) < len(literal):
                return (length, False) if literal.startswith(
                    remaining) else None
            return (position + len(literal), True) if remaining.startswith(
                literal) else None
        if char == "-" or "0" <= char <= "9":
            return parse_number(position)
        return None

    value = parse_value(0)
    if value is None:
        return False
    position, complete = value
    if not complete:
        return position == length
    return skip_whitespace(position) == length


def _matches_json_type(value: object, json_type: str) -> bool:
    """Return whether value matches the specified JSON type."""
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "null":
        return value is None
    return False


def _is_plain_json_string_content(text: str) -> bool:
    """Whether text ends inside a JSON string, outside an escape sequence."""
    in_string = False
    escaped = False
    unicode_digits = 0
    for char in text:
        if not in_string:
            if char == '"':
                in_string = True
        elif unicode_digits:
            unicode_digits -= 1
        elif escaped:
            escaped = False
            if char == "u":
                unicode_digits = 4
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = False
    return in_string and not escaped and unicode_digits == 0


def _has_only_parameters_top_level(candidate: str) -> bool:
    """Reject extra top-level fields after the parameters object."""
    parameters_marker = '", "parameters": {'
    marker_start = candidate.find(parameters_marker)
    if marker_start == -1:
        return True

    depth = 1
    in_string = False
    escaped = False
    for position in range(marker_start + len(
            parameters_marker), len(candidate)):
        char = candidate[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                remaining = candidate[position + 1:].strip()
                return not remaining or (
                    remaining.startswith("}") and not remaining[1:].strip()
                )
    return True


class JSONConstrainedDecoder:
    """Constrain output to a valid JSON call
    envelope and a registered function name."""

    def __init__(self, vocab: Vocabulary, functions:
                 List[FunctionDefinition]) -> None:
        self.vocab = vocab
        self.functions = functions
        self.function_names = [fn.name for fn in functions]
        self.vocab_tokens = vocab.get_all_tokens()
        self.tokens_by_first_char: dict[str, list[tuple[int, str]]] = {}
        for token_id, token_str in self.vocab_tokens:
            if token_str:
                self.tokens_by_first_char.setdefault(
                    token_str[0], []).append((token_id, token_str))

    def normalize_call(self, call_data: object) -> dict[str, Any] | None:
        """Validate a call and normalize JSON-encoded
        numeric strings to numbers."""
        if not isinstance(call_data, dict) or set(
                call_data) != {"name", "parameters"}:
            return None

        function = next(
            (fn for fn in self.functions if fn.name == call_data["name"]),
            None,
        )
        parameters = call_data["parameters"]
        if function is None or not isinstance(parameters, dict):
            return None
        if set(parameters) != set(function.parameters):
            return None

        normalized_parameters: dict[str, Any] = {}
        for name, definition in function.parameters.items():
            value = parameters[name]
            if definition.type in {"number", "integer"
                                   } and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return None
            if not _matches_json_type(value, definition.type):
                return None
            if isinstance(value, float) and not math.isfinite(value):
                return None
            normalized_parameters[name] = value
        return {"name": function.name, "parameters": normalized_parameters}

    def is_valid_call(self, call_data: object) -> bool:
        """Check whether a call can be normalized
        to its registered parameter schema."""
        return self.normalize_call(call_data) is not None

    def _is_valid_prefix(self, generated_text: str,
                         next_token_str: str) -> bool:
        """
        Determines if adding `next_token_str`
        keeps the JSON structurally sound.
        This is a simplified state machine. A production system (like outlines)
        builds a full Regex Finite State Machine (FSM). For this project, we
        simulate state by checking the current string's structure.
        """
        candidate = generated_text + next_token_str

        # State 1: Beginning of JSON up to function name
        template_start = '{"name": "'
        if len(candidate) <= len(template_start):
            return template_start.startswith(candidate)

        # State 2: Match a complete function name before
        # allowing the transition to parameters.
        name_part = candidate[len(template_start):]
        name_end = name_part.find('"')
        if name_end == -1:
            return any(fn.startswith(name_part) for fn in self.function_names)

        if name_part[:name_end] not in self.function_names:
            return False

        transition_template = '", "parameters": {'
        after_name = name_part[name_end:]
        if not after_name.startswith(transition_template):
            return transition_template.startswith(after_name)

        if not _has_only_parameters_top_level(candidate):
            return False
        if not _is_valid_json_prefix(candidate):
            return False
        try:
            call_data, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            return True
        if candidate[end:].strip():
            return False
        return self.normalize_call(call_data) is not None

    def apply_mask(self, logits: List[float], generated_text: str
                   ) -> npt.NDArray[np.float64]:
        """
        Mask tokens outside the valid JSON call format or tokenizer vocabulary.
        """
        np_logits = np.array(logits, dtype=np.float64)
        masked_logits = np.full(np_logits.shape, -np.inf, dtype=np.float64)

        plain_string_content = _is_plain_json_string_content(generated_text)
        if plain_string_content:
            candidate_tokens = self.vocab_tokens
        else:
            valid_first_chars = {
                char
                for char in self.tokens_by_first_char
                if self._is_valid_prefix(generated_text, char)
            }
            candidate_tokens = [
                token
                for char in valid_first_chars
                for token in self.tokens_by_first_char[char]
            ]

        for token_id, token_str in candidate_tokens:
            if token_id >= len(np_logits):
                continue
            if plain_string_content and '"' not in token_str and (
                    "\\") not in token_str:
                if all(ord(char) >= 0x20 for char in token_str):
                    masked_logits[token_id] = np_logits[token_id]
                    continue
            if self._is_valid_prefix(generated_text, token_str):
                masked_logits[token_id] = np_logits[token_id]

        return masked_logits
