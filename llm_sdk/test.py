from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

prompt = "The capital of France is"

input_ids = model.encode(prompt)

print("Input IDs:", input_ids)
print("Shape:", input_ids.shape)

logits = model.get_logits_from_input_ids(input_ids[0].tolist())

print("Vocabulary size:", len(logits))

# Find the 10 most likely next tokens
top_k = 10
top_ids = sorted(
    range(len(logits)),
    key=lambda i: logits[i],
    reverse=True
)[:top_k]

for token_id in top_ids:
    token = model.decode([token_id])
    print(f"{token_id:>6}: {token!r}  logit={logits[token_id]:.3f}")
