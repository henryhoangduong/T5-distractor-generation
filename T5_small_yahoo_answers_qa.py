import os

import evaluate
import nltk
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5Tokenizer)

import wandb

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))
login(os.getenv("HF_TOKEN"))

DATA_NAME = "nfL6/yahoo_answers_qa"
MODEL_NAME = "google/flan-t5-base"
L_RATE = 3e-4
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 3

nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


yahoo_answers_qa = load_dataset(DATA_NAME, trust_remote_code=True)
yahoo_answers_qa = yahoo_answers_qa["train"].train_test_split(test_size=0.3)


prefix = "Please answer this question: "


def preprocess_function(examples):
    """Add prefix to the sentences, tokenize the text, and set the labels"""
    # The "inputs" are the tokenized answer:
    inputs = [prefix + doc for doc in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # The "labels" are the tokenized outputs:
    labels = tokenizer(text_target=examples["answer"],
                       max_length=512,
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels, use_stemmer=True)

    return result


tokenized_dataset = yahoo_answers_qa.map(preprocess_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=L_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=SAVE_TOTAL_LIM,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    push_to_hub=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.push_to_hub(
    "henryhoangduong/T5_small_distractor_generation_yahoo_answers_qa_gpu")
