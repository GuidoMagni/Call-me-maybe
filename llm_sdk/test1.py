from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

prompt = "The capital of France is"

result = model.generate(
    prompt,
    max_new_tokens=30,
    temperature=0.7,
)

print("PROMPT:")
print(prompt)

print("\nOUTPUT:")
print(result)