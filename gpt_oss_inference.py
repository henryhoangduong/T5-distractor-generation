import ollama
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import torch
torch.cuda.empty_cache()
# Initialize evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Load RACE validation subset
# smaller slice for testing
dataset = load_dataset("race", "all")
dataset = dataset["test"]

responses = []

for example in tqdm(dataset, desc="Generating distractors"):
    question = example["question"]
    correct_label = example["answer"]
    context = example["article"]
    options = example["options"]
    label_index = ord(correct_label) - ord("A")
    correct_text = options[label_index]

    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct_text}\n"
        "Generate three plausible but incorrect answer choices (distractors). "
        "Separate them with semicolons."
    )

    resp = ollama.chat(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for MCQ creation."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.7}
    )

    generated_str = resp["message"]["content"].strip()
    responses.append({"prediction": generated_str, "reference": correct_text})
    print("===========================")
    print("generated_str: ", generated_str)
    print("references: ", [correct_text])

    bleu.add(prediction=generated_str, references=[correct_text])
    rouge.add(prediction=generated_str, reference=correct_text)

# Compute final metrics
results_bleu = bleu.compute()
results_rouge = rouge.compute()

print("BLEU:", results_bleu)
print("ROUGE:", results_rouge)
