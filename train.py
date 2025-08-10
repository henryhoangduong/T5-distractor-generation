from transformers import AutoTokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
)
import os

from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import numpy as np
import evaluate
import torch
import wandb
from dotenv import load_dotenv
import wandb
import math
load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"))
login(os.getenv("HF_TOKEN"))

dataset = load_dataset("ehovy/race", "all")

MODEL = 't5-small'                  # the Model name here
BATCH_SIZE = 20                     # The batch size here
NUM_PROCS = 10                      # The num of proccess here
EPOCHS = 1                          # The number of training Epochs here
OUT_DIR = 'results_t5small'         # the outout directory name here
MAX_LENGTH = 512                    # the max length of the sequence here

print(f"Dataset type: {type(dataset)}")
print(f"Dataset length: {len(dataset)}")
print(f"Dataset keys: {dataset.keys()}")

train_dataset = dataset["train"]
half_len = math.floor(len(train_dataset) / 2)
train_dataset = dataset["train"].select(range(500))
eval_dataset = dataset['validation']
test_dataset = dataset['test']

print(f"Train section type: {type(train_dataset)}")
print(f"Train section length: {len(train_dataset)}")

print(train_dataset[0].keys())

tokenizer = AutoTokenizer.from_pretrained(MODEL)


tokenizer = AutoTokenizer.from_pretrained(MODEL)
MAX_LENGTH = 512


def preprocess_function(examples):
    inputs = []
    targets = []

    for article, question, options, answer in zip(
        examples["article"], examples["question"], examples["options"], examples["answer"]
    ):
        correct_index = ord(answer) - ord("A")
        correct_answer = options[correct_index]
        distractors = [opt for i, opt in enumerate(
            options) if i != correct_index]

        for distractor in distractors:
            prompt = f"generate distractor: {question} answer: {correct_answer} context: {article}"
            inputs.append(prompt)
            targets.append(distractor)

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    label_tokens = tokenizer(
        text_target=targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = label_tokens["input_ids"]

    return model_inputs


tokenized_train_dataset = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")

total_trainable_params = sum(p.numel()
                             for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if predictions.ndim > 2:  # logits case
        predictions = np.argmax(predictions, axis=-1)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds,
                           references=decoded_labels, use_stemmer=True)

    pred_tokens = [pred.split() for pred in decoded_preds]
    label_tokens = [[label.split()] for label in decoded_labels]
    bleu_result = bleu.compute(
        predictions=pred_tokens, references=label_tokens)
    result["bleu"] = round(bleu_result["bleu"], 4)

    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=OUT_DIR,
    logging_steps=50,
    fp16=True,
    eval_strategy='epoch',
    eval_steps=1000,
    save_strategy='epoch',
    save_steps=1000,
    save_total_limit=3,
    learning_rate=5e-4,
    dataloader_num_workers=8,
    report_to='wandb',
    predict_with_generate=True,
    push_to_hub=True,
    remove_unused_columns=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.push_to_hub("henryhoangduong/T5-small-distractor-generation")
