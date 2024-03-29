import sys
sys.path.append("[PATH]/Clinical-LLM/clinical-LLM/Finetuning/")

import transformers as ts
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from utils import DataCollatorForCompletionOnlyLM
from config import *

import pickle

import os
import argparse
import numpy as np
import transformers as ts
from datasets import load_from_disk
import torch
import numpy as np

from huggingface_hub import HfFolder

from accelerate import Accelerator

from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

import pynvml

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the already processed dataset.")
    parser.add_argument(
        "--repository_id", type=str, default=None, help="Hugging Face Repository id for uploading models"
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Path to deepspeed config file.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    args = parser.parse_known_args()
    return args

def evaluation_function(args):
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)

    labels = []
    # types = []

    for row in dataset:
        labels.append(row["output"])
        # types.append(row["type"])

    tokenizer = AutoTokenizer.from_pretrained("[PATH]/Clinical-LLM/clinical-LLM/Llama-2-13b-chat-hf/")
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    print("PAD TOKEN = " + tokenizer.pad_token)
    print("PAD TOKEN ID = " + str(tokenizer.pad_token_id))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="balanced",
        torch_dtype=torch.float16,
        # trust_remote_code=True,
    )

    # model.generation_config.pad_token_id = tokenizer.pad_token_id

    print(model)

    # dataset = dataset.select(range(32))

    def mapping_function(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Instruction: {example['instruction'][i]}\n ### Input: {example['input'][i]}\n ### Output:\n"
            output_texts.append(text)
        return tokenizer(output_texts)

    dataset = dataset.map(mapping_function, batched=True, remove_columns=dataset.column_names)

    # print(dataset)
    # print(dataset[0])

    collator = ts.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    output_dir = args.model_id.strip()[:-1].split("/")[-1]
    training_args = ts.Seq2SeqTrainingArguments(
            output_dir=f"{output_dir}/tests",
            predict_with_generate=True,
            generation_max_length=args.generation_max_length,
            generation_num_beams=4,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=True,  # T5 overflows with fp16
            learning_rate=args.lr,
            warmup_steps=0,
            num_train_epochs=args.epochs,
            gradient_checkpointing=args.gradient_checkpointing,
            # deepspeed=args.deepspeed,
    )

    trainer = ts.Seq2SeqTrainer(
        model,
        args=training_args,
        train_dataset=dataset.select(range(4)),
        data_collator=collator,
        callbacks=[ts.ProgressCallback()]
    )

    outputs = trainer.predict(dataset)
    predictions = np.array(outputs.predictions)

    print(predictions.shape)

    save_path = os.path.join(args.dataset_path, "outputs")

    print(save_path)

    with open(os.path.join(save_path, "generated-13-final.pickle"), "wb") as f:   #Pickling
        pickle.dump(predictions, f)

    with open(os.path.join(save_path, "real-13-final.pickle"), "wb") as f:   #Pickling
        pickle.dump(labels, f)

    # with open(os.path.join(save_path, "types.pickle"), "wb") as f:   #Pickling
    #     pickle.dump(types, f)

    # print(tokenizer.batch_decode(list(outputs.predictions), skip_special_tokens=True))

def main():
    args, _ = parse_arge()
    evaluation_function(args)

if __name__ == "__main__":
    main()