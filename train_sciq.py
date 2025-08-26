import math
import os

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

import wandb

load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"))
login(os.getenv("HF_TOKEN"))


dataset = load_dataset("allenai/sciq")

MODEL = 't5-small'
BATCH_SIZE = 20
NUM_PROCS = 10
EPOCHS = 3
OUT_DIR = 'results_t5small'
MAX_LENGTH = 512
