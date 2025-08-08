import ollama
from datasets import load_dataset
import evaluate
from tqdm import tqdm
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
    correct = example["answer"]
    context = example["article"]  # in RACE dataset, "article" is the passage

    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Correct Answer: {correct}\n"
        "Generate three plausible but incorrect answer choices (distractors). "
        "Separate them with semicolons."
    )

    # Call GPT-OSS 20B via Ollama
    resp = ollama.chat(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for MCQ creation."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.7}
    )

    generated = resp["message"]["content"]
    responses.append({"prediction": generated, "reference": correct})

    # Update BLEU/ROUGE (this is illustrative — normally you'd have gold distractors)
    bleu.add(prediction=generated.split(), references=[[correct.split()]])
    rouge.add(prediction=generated, reference=correct)

# Compute final metrics
results_bleu = bleu.compute()
results_rouge = rouge.compute()

print("BLEU:", results_bleu)
print("ROUGE:", results_rouge)
