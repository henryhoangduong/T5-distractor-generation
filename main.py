from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

dataset = load_dataset("race", "middle")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


def tokenize(batch):
    return tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=512)


def tokenize_labels(batch):
    return tokenize(batch["input_text"], padding="max_length", truncation=True, max_length=64)


def preprocess(example):
    correct_idx = ord(example["answer"]) - ord("A")
    correct = example["options"][correct_idx]
    distractors = [opt for i, opt in enumerate(
        example["options"]) if i != correct_idx]

    input_text = f"generate distractors: question: {example['question']} context: {example['article']} answer: {correct}"
    target_text = " | ".join(distractors)

    return {"input_test": input_text, "target_text": target_text}


processed = dataset.map(preprocess)

tokenized_inputs = processed.map(tokenize, batched=True)
tokenized_targets = processed.map(tokenize_labels, batched=True)

tokenized_inputs = tokenized_inputs.map(
    lambda x: {"labels": x["input_ids"]}, batched=True)

training_args = TrainingArguments(
    output_dir="./t5-distractor",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_inputs["train"]
)

trainer.train()
tokenized_inputs["train"] = tokenized_inputs["train"].select(range(1000))
