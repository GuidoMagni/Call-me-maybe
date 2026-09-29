from llm_sdk import Small_LLM_Model

tests = [
    "The capital of France is",
    "2 + 2 =",
    "Python is a programming language used for",
    "Once upon a time",
    "Explain gravity in one sentence:",
]

model = Small_LLM_Model()

for prompt in tests:
    print("=" * 60)
    print("PROMPT:", prompt)

    output = model.generate(
        prompt,
        max_new_tokens=40,
        temperature=0.0,
        do_sample=False,
    )

    print("OUTPUT:", output)