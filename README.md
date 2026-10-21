*This project has been created as part of the 42 curriculum by gumagni.*

# Call me maybe

## Description

Call me maybe translates natural-language requests into structured function calls. It loads function definitions and prompts from JSON files, asks the Qwen3-0.6B language model to select a function and produce its arguments, and writes one validated JSON result per prompt.

The project focuses on constrained decoding: instead of trusting the model to produce valid JSON on its own, it masks invalid next-token choices during generation and validates the completed call against the registered function schema.

## Instructions

### Requirements

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/)
- Internet access on the first run to download the model and tokenizer from Hugging Face
- Enough memory and compute for Qwen3-0.6B; inference uses a supported GPU when available and otherwise falls back to CPU

### Install and run

From the repository root:

```sh
uv sync
uv run python -m src
```

The default inputs are `data/input/functions_definition.json` and `data/input/function_calling_tests.json`. The default output is `data/output/function_calling_results.json`; the program creates `data/output/` when needed. Generated output is not part of the submission and should not be committed.

Input and output paths can be overridden:

```sh
uv run python -m src \
  -functions_definition data/input/functions_definition.json \
  -input data/input/function_calling_tests.json \
  -output data/output/function_calling_results.json
```

The output is a JSON array. Each entry contains exactly `prompt`, `name`, and `parameters`; parameter names and JSON types are checked against the selected function definition. Missing files, invalid JSON, and invalid generated calls are reported as errors.

## Resources

- [Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [Let's build the GPT Tokenizer — Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE)

### AI use

AI assistance was used to draft the finite-state-machine concepts in the constrained decoder and the Pydantic validation boilerplate. AI assistance was also used to revise this README. Generated material must be understood and verified by the project author.

## Algorithm and design decisions

1. The application reads the function definitions and prompt list from JSON, validates their records with Pydantic, and builds a prompt containing the available function names, descriptions, parameter names, and types.
2. The model encodes that prompt. At each generation step, the decoder examines candidate vocabulary tokens and masks invalid choices by setting their logits to negative infinity. Generation selects the highest-logit token that remains.
3. The decoder requires the output to follow the `{"name": "...", "parameters": {...}}` envelope, limits the name to a registered function, and checks that the text remains a possible JSON prefix. Its prefix parser accounts for JSON strings and escapes, nested objects and arrays, literals, and numbers.
4. Once a complete JSON object is available, the decoder rejects extra or missing top-level keys, unknown function names, missing or extra parameters, and values that do not match the selected function's parameter types. Numeric strings are parsed as JSON numbers before the final type check.
5. The application adds the original prompt to each accepted call and writes the result list as formatted JSON.

The decoder uses a purpose-built JSON-prefix validator rather than a general-purpose grammar library. The generation loop uses greedy selection, and it uses the SDK's cached-logit method when available to avoid recomputing the full sequence on every step. The model prompt helps the model choose the appropriate function and values, but the decoder and final schema check—not prompting alone—enforce the output structure.

## Performance and reliability

The implementation is designed to keep generated output parseable and schema-compliant by masking structurally invalid prefixes and validating completed calls. This depends on the vocabulary-to-text mapping accurately representing the model's tokens.

No repeatable accuracy or elapsed-time benchmark is currently included in the repository. Therefore, the assignment's targets of at least 90% correct function selection and argument extraction and processing within five minutes have not been demonstrated. Function selection and argument extraction remain model-dependent, even when the output structure is constrained. Measure accuracy against labeled prompts and record runtime on the evaluation hardware before making quantitative performance claims.

## Challenges

- **Keeping partial output valid:** A token must be checked in context, not just as an isolated character. The decoder tests candidate text against the required envelope and JSON-prefix rules before allowing it.
- **Avoiding extra fields:** Valid JSON can still violate the required call format. The decoder checks the exact top-level keys, function-specific parameter names, and parameter types after parsing.
- **Handling tokenized text:** Vocabulary entries are converted to text before prefix checks, including common space and newline markers. Tokenizer-specific representations should be verified if the model or tokenizer changes.

## Testing strategy

The repository includes sample prompts and function schemas under `data/input/`, but no automated test suite or recorded benchmark is currently included. For a manual check:

1. Install dependencies and run the default command shown above.
2. Confirm that the output file is created and can be parsed as JSON.
3. For every result, check that the prompt is preserved, the function name exists in the definitions, all required parameters are present, and each value has the declared JSON type.
4. Compare each selected function and extracted value with the expected answer to assess semantic accuracy; record total runtime separately.
5. Repeat with changed input files, including strings with punctuation or escapes, different numeric values, ambiguous requests, and functions with different parameter sets.

## Example

Given the prompt `What is the sum of 40 and 2?` and an `fn_add_numbers` definition with numeric parameters `a` and `b`, a valid result has this shape:

```json
[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 40,
      "b": 2
    }
  }
]
```
