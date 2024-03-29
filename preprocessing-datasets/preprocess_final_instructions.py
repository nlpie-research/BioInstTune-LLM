import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"

path = os.path.join(current_path, "instruction-datasets")
save_path = os.path.join(current_path, "instruction-datasets", "avicenna-instructions")

datasets = [
    ds.load_from_disk(os.path.join(path, "ner", "ner-instruction")),
    ds.load_from_disk(os.path.join(path, "cls", "hoc")),
    ds.load_from_disk(os.path.join(path, "nli", "MedNLI")),
    # ds.load_from_disk(os.path.join(path, "re", "gad")),
    ds.load_from_disk(os.path.join(path, "re", "i2b2-2010")),
]

dataset = ds.concatenate_datasets(datasets)
print(dataset)

dataset = dataset.shuffle()

print(dataset)

def show_in_instruction_mode(index):
    print("### Instruction: \n" + dataset[index]["instruction"] + "\n")
    print("### Input: \n" + dataset[index]["input"] + "\n")
    print("### Output: \n" + dataset[index]["output"] + "\n")

show_in_instruction_mode(10)
print()
show_in_instruction_mode(100)

dataset.save_to_disk(save_path)