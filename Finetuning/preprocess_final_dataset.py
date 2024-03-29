import os
import json

import datasets as ds

import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM"

instruction_path = os.path.join(current_path, "instruction-datasets")

avicenna = ds.load_from_disk(os.path.join(instruction_path, "avicenna-instructions"))
chatdoctor = ds.load_from_disk(os.path.join(instruction_path, "chatdoctor"))
pmc_llama = ds.load_from_disk(os.path.join(instruction_path, "pmc-llama-v2"))

dataset = ds.concatenate_datasets([avicenna, chatdoctor, pmc_llama])
dataset = dataset.shuffle()

print(dataset)
dataset.save_to_disk(os.path.join(instruction_path, "final-avicenna-instructions"))

def show_in_instruction_mode(index):
    print("### Instruction: \n" + dataset[index]["instruction"] + "\n")
    print("### Input: \n" + dataset[index]["input"] + "\n")
    print("### Output: \n" + dataset[index]["output"] + "\n")

show_in_instruction_mode(10)
print()
show_in_instruction_mode(100)