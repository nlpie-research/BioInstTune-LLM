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
from collections import Counter

from load_dataset_chunk import get_dataset

metric = load_metric("f1")

load_path = "[PATH]/Clinical-LLM/clinical-LLM/instruction-datasets/eval/outputs/"

tokenizer = ts.AutoTokenizer.from_pretrained("[PATH]/Clinical-LLM/clinical-LLM/Llama-2-13b-chat-hf")
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "left"

datasetTag = "i2b2-2010"

generated, groundtruths = get_dataset(os.path.join(load_path, "generated-13-final.pickle"), os.path.join(load_path, "groundtruth-13-final.pickle"), os.path.join(load_path, "types-13-final.pickle"), datasetTag)

allLabels = set(groundtruths)

def get_labels(inputs):
    count = 0
    labels = []
    for input in inputs:
        output = input.split("### Output:")[1].strip().replace("\n","").strip()
        for label in allLabels:
            if label in output:
                if label.strip() != output.strip():
                    count += 1
                
                labels.append(label)
                break
    
    return labels, count

generated, count = get_labels(generated)
print("Count is: " + str(count))

print(set(generated))

labels_to_index = {
    label: index for index, label in enumerate(set(groundtruths))
}

groundtruths = groundtruths[:-1]

print(generated[:10])
print(groundtruths[:10])

final_generated = []
final_ground = []

count = 0

for index, gen in enumerate(generated):
    if gen == '' or gen == 'PCP':
        count += 1
        final_generated.append(labels_to_index["No Relations"])
    else:
        final_generated.append(labels_to_index[gen])

for ground in groundtruths:
    final_ground.append(labels_to_index[ground])

print("Counts: " + str(count))
gen_count = Counter(final_generated)
ground_count = Counter(final_ground)

print(dict(sorted(gen_count.items())))
print(dict(sorted(ground_count.items())))

print(metric.compute(predictions=final_generated, references=final_ground, average="micro"))