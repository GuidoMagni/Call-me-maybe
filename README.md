*This project has been created as part of the 42 curriculum by gumagni.*

## Description
This project implements a Function Calling tool bridging Large Language Models and external systems. It demonstrates how to enforce structured outputs (JSON) from an LLM by utilizing constrained decoding—a technique that restricts the model's vocabulary during generation based on a predefined grammar schema.

## Instructions
Ensure you have `uv` installed.
1. `make install` to synchronize and lock dependencies.
2. `make lint-strict` to check typing and formatting.
3. `make run` to execute the data processing pipeline.

## Resources
- [Let's build the GPT Tokenizer (Andrej Karpathy)](https://www.youtube.com/watch?v=zduSFxRajkE)
- AI Usage: AI was utilized to draft the finite state machine concepts in the constrained decoder logic and to establish the Pydantic data validation boilerplate.

## Design Decisions & Algorithm Explanation
**Constrained Decoding via Prefix Matching:**
Instead of relying on prompt engineering, this solution uses a greedy state-machine inside `JSONConstrainedDecoder`. At every token generation step:
1. We simulate appending the token to our `generated_text`.
2. We evaluate if this candidate string breaks the required JSON structure (`{"name": "[allowed_function]", "parameters": {...}}`).
3. If it breaks the structure, we set the token's probability logit to negative infinity (`-np.inf`), physically preventing the model from selecting it.