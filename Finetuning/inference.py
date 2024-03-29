import gc
import math
import os
import time
from argparse import ArgumentParser

import torch
import torch.distributed as dist

import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
parser.add_argument("--cpu_offload", action="store_true", help="whether to activate CPU offload")
parser.add_argument("--nvme_offload_path", help="whether to activate NVME offload and the path on nvme")
args = parser.parse_args()

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = args.name

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.hidden_size

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size

ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}

dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

model = AutoModelForCausalLM.from_pretrained(model_name)

ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

rank = torch.distributed.get_rank()

text_in = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"

# if rank == 0:
#     text_in = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
# elif rank == 1:
#     text_in = "Is this review positive or negative? Review: this is the worst restaurant ever"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, synced_gpus=True)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"rank{rank}:\n   in={text_in}\n  out={text_out}")