from datasets import load_dataset
import evaluate
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config, AutoTokenizer
torch.cuda.empty_cache()

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


dataset = load_dataset("race", "all")["validation"]

# Example row
example = dataset[0]
article = example["article"]
question = example["question"]
options = example["options"]
answer_letter = example["answer"]

# Map letter to actual text
letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
correct_answer = options[letter_to_index[answer_letter]]

prompt = f"""
You are a multiple-choice question creator.
Given the following article and correct answer, create 3 plausible but incorrect distractor options.

Article:
{article}

Question:
{question}

Correct Answer:
{correct_answer}

Return only the distractors as a list, without explanations.
"""


quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b", **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")


messages = [
    {"role": "user", "content": ""},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

# Generate
output_ids = model.generate(input_ids, max_new_tokens=256)
response = tokenizer.decode(
    output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

print("Correct answer:", correct_answer)
print("Generated distractors:\n", response.strip())


# for example in tqdm(dataset, desc="Generating distractors"):
#     question = example["question"]
#     correct_label = example["answer"]
#     context = example["article"]
#     options = example["options"]
#     label_index = ord(correct_label) - ord("A")
#     correct_text = options[label_index]

#     prompt = (
#         f"Context: {context}\n"
#         f"Question: {question}\n"
#         f"Correct Answer: {correct_text}\n"
#         "Generate three plausible but incorrect answer choices (distractors). "
#         "Separate them with semicolons."
#     )

#     resp = ollama.chat(
#         model="gpt-oss:20b",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant for MCQ creation."},
#             {"role": "user", "content": prompt}
#         ],
#         options={"temperature": 0.7}
#     )

#     generated_str = resp["message"]["content"].strip()
#     responses.append({"prediction": generated_str, "reference": correct_text})
#     print("===========================")
#     print("generated_str: ", generated_str)
#     print("references: ", [correct_text])

#     bleu.add(prediction=generated_str, references=[correct_text])
#     rouge.add(prediction=generated_str, reference=correct_text)

# # Compute final metrics
# results_bleu = bleu.compute()
# results_rouge = rouge.compute()

# print("BLEU:", results_bleu)
# print("ROUGE:", results_rouge)
