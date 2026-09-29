from llm_sdk import Small_LLM_Model

tests = [
    "what is the capital of italy",
    "2 + 2 =",
    "Python is a programming language used for",
    "Once upon a time",
    "Explain gravity in one sentence:",
    "Who made Qwen3-0.6B",
    "do you know that you are Qwen",
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