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
from transformers.trainer_utils import get_last_checkpoint

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

def training_function(args):
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)

    print(dataset)
    # print({"": Accelerator().process_index})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # tokenizer.model_max_length = args.generation_max_length
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    print("PAD TOKEN = " + tokenizer.pad_token)

    # dataset = dataset.select(range(1000))

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Instruction: {example['instruction'][i]}\n ### Input: {example['input'][i]}\n ### Output: {example['output'][i]} \n{tokenizer.eos_token}"
            # text = f"### Instruction: {example['instruction'][i]}\n ### Input: {example['input'][i]}\n ### Output: {example['output'][i]}\n{tokenizer.eos_token}"
            # text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    response_template = "### Output:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    output_dir = args.model_id.strip()[:-1].split("/")[-1]
    training_args = ts.TrainingArguments(
            output_dir=f"{output_dir}/checkpoints",
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=True,  # T5 overflows with fp16
            bf16=False,  # Use BF16 if available
            learning_rate=args.lr,
            warmup_steps=1000,
            num_train_epochs=args.epochs,
            deepspeed=args.deepspeed,
            gradient_checkpointing=args.gradient_checkpointing,
            # logging & evaluation strategies
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        # device_map={'':torch.cuda.current_device()},
        # device_map="balanced", # Using it for the zero stage 2
        # low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # print(model)

    with open(os.path.join(output_dir,  "logs.txt"), "w") as f:
        f.write("")

    class CustomCallback(ts.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                # print(logs)
                with open(os.path.join(output_dir,  "logs.txt"), "a+") as f:
                    f.write(str(logs) + "\n")

    trainer = SFTTrainer(
        model,
        max_seq_length=args.generation_max_length,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        callbacks=[ts.ProgressCallback(), CustomCallback()]
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/checkpoints/final")


def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()