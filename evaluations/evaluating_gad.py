import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import datasets as ds
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers as ts
import pickle
import numpy as np
from datasets import load_metric
import seqeval

metric = load_metric("accuracy")

load_path = "[PATH]/Clinical-LLM/clinical-LLM/instruction-datasets/re/gad-eval/outputs/"

tokenizer = ts.AutoTokenizer.from_pretrained("[PATH]/Clinical-LLM/clinical-LLM/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"

with open(os.path.join(load_path, "real.pickle"), mode="rb") as f:
    groundtruths = pickle.load(f)

with open(os.path.join(load_path, "generated.pickle"), mode="rb") as f:
    generated = pickle.load(f)

generated = np.array(generated)
generated[generated == -100] = tokenizer.pad_token_id

decoded_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)

def get_labels(inputs):
    return [input.split("### Output:")[-1].strip().replace("\n","").strip() for input in inputs]

generated = get_labels(decoded_outputs)

print(set(generated))
print(set(groundtruths))

print(generated[:8])
print(groundtruths[:8])

# label_to_index = {
#     label: index for index, label in enumerate(set(groundtruths))
# }

# def format_labels(inputs):
#     return [label_to_index[input] for input in inputs]

# print(metric.compute(predictions=format_labels(generated), references=format_labels(groundtruths)))
