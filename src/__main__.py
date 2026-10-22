"""Main entry point for the function calling application."""
import argparse
import json
import os
import sys
import numpy as np
from typing import Any, Dict, List
from pydantic import ValidationError

# Adjust imports according to your local SDK location
try:
    from llm_sdk import Small_LLM_Model
except ImportError:
    print("Error: llm_sdk not found. Please ensure it is"
          " in the project root.", file=sys.stderr)
    sys.exit(1)

from src.models import FunctionDefinition, PromptInput, FunctionCallOutput
from src.vocab import Vocabulary
from src.decoder import JSONConstrainedDecoder


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Function Calling"
                                                 " Constrained Decoder")
    parser.add_argument('-functions_definition', type=str,
                        default='data/input/functions_definition.json')
    parser.add_argument('-input', type=str,
                        default='data/input/function_calling_tests.json')
    parser.add_argument('-output', type=str,
                        default='data/output/function_calling_results.json')
    return parser.parse_args()


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Safely load a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON root must be an array.")
            return data
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def generate_function_call(
    model: Any,
    decoder: JSONConstrainedDecoder,
    prompt: str
) -> Dict[str, Any]:
    """Generate a function call matching one of the
    registered function schemas."""
    function_schemas = [
        {
            "name": function.name,
            "description": function.description,
            "parameters": {
                name: parameter.type
                for name, parameter in function.parameters.items()
            },
        }
        for function in decoder.functions
    ]
    system_prompt = (
        "Select the single best matching function "
        "from the available functions below. "
        "Return only one JSON object with exactly the "
        "keys \"name\" and \"parameters\". "
        "The name must exactly match a listed function. "
        "Include every listed parameter "
        "for that function, use the specified JSON "
        "types, and do not add other keys. "
        "A number parameter must be a JSON "
        "number, not a quoted string. "
        "A string parameter must be quoted. Do "
        "not repeat the function name or parameters "
        "after the JSON object. Use the user's values "
        "faithfully; do not invent values.\n"
        f"Available functions:\n{json.dumps(
            function_schemas, ensure_ascii=False)}\n"
        "For regex replacements, match the target "
        "in the current request: replacing "
        "the word 'cat' with 'dog' uses regex "
        "'cat' and replacement 'dog'; replacing "
        "all vowels with asterisks uses regex "
        "'[aeiouAEIOU]' and replacement '*'. "
        "Do not reuse an example value when it "
        "does not match the request.\n"
        f"User request: {prompt}\n"
        "Function call JSON:"
    )

    input_ids: List[int] = model.encode(system_prompt)[0].tolist()
    generated_text = ""
    past_key_values: Any = None
    cached_logits = getattr(
        model, "get_logits_from_input_ids_with_cache", None)

    max_tokens = 200  # Safety limit to prevent infinite loops

    for _ in range(max_tokens):
        if callable(cached_logits):
            logits, past_key_values = cached_logits(input_ids, past_key_values)
        else:
            logits = model.get_logits_from_input_ids(input_ids)

        # --- Core Constrained Decoding Step ---
        masked_logits = decoder.apply_mask(logits, generated_text)
        if not np.isfinite(masked_logits).any():
            raise RuntimeError("No valid next token while generating "
                               f"a call for prompt: {prompt}")

        # Select highest probability valid token
        next_token_id = int(np.argmax(masked_logits))
        input_ids.append(next_token_id)

        # Decode token and append (safely utilizing
        # vocab if decode isn't provided)
        next_token_str = decoder.vocab.id_to_token.get(next_token_id, "")
        generated_text += next_token_str

        try:
            call_data, end = json.JSONDecoder().raw_decode(generated_text)
        except json.JSONDecodeError:
            continue
        if not generated_text[end:].strip() and isinstance(call_data, dict):
            normalized_call = decoder.normalize_call(call_data)
            if normalized_call is not None:
                return normalized_call
            raise RuntimeError(
                f"Generated function call does not match a registered schema "
                f"for prompt {prompt!r}: {call_data!r}"
            )

    raise RuntimeError(
        "Failed to generate a complete JSON "
        f"function call for prompt {prompt!r} "
        f"within {max_tokens} tokens. Partial output: {generated_text!r}"
    )


def main() -> None:
    """Main application loop."""
    args = parse_arguments()

    # 1. Load Data
    funcs_raw = load_json_file(args.functions_definition)
    prompts_raw = load_json_file(args.input)

    # 2. Validate Data with Pydantic
    try:
        functions = [
            FunctionDefinition.model_validate(function)
            for function in funcs_raw
        ]
        prompts = [
            PromptInput.model_validate(prompt)
            for prompt in prompts_raw
        ]
    except ValidationError as e:
        print(f"Error validating input data: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Initialize Model and SDK
    model = Small_LLM_Model()
    vocab = Vocabulary(model.get_path_to_vocab_file())
    decoder = JSONConstrainedDecoder(vocab, functions)

    results: List[Dict[str, Any]] = []

    # 4. Process each prompt
    for p in prompts:
        print(f"Processing prompt: {p.prompt}")
        call_data = generate_function_call(model, decoder, p.prompt)

        output = FunctionCallOutput(
            prompt=p.prompt,
            name=call_data.get("name", "unknown"),
            parameters=call_data.get("parameters", {})
        )
        results.append(output.model_dump())

    # 5. Output results
    output_directory = os.path.dirname(args.output) or "."
    try:
        os.makedirs(output_directory, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
    except OSError as e:
        print(f"Error writing output file {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Success! Output written to {args.output}")


if __name__ == "__main__":
    main()
