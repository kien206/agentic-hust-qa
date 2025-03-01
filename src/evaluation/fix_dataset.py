import json

# Load your JSON file
with open("../data/eval/test_dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Assuming the structure of input is {'question': List[str], 'response': List[str]}
questions = data.get("question", [])
contexts = data.get("contexts", [])
answer = data.get("answer", [])
ground_truth = data.get("ground_truths", [])

# Create the new format
output = []
for question, contexts, answer, ground_truth in zip(
    questions, contexts, answer, ground_truth
):
    obj = {
        "user_input": question,
        "retrieved_contexts": contexts,
        "response": answer,
        "reference": ground_truth,
    }
    output.append(obj)

# Save to a file or print
data = {"version": "1.0.0", "data": output}
with open("output.json", "w", encoding="utf-8") as file:
    file.write(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )  # Write in newline-delimited JSON format

print("Conversion completed.")
