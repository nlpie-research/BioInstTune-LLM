import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"
dataset_path = os.path.join(current_path, "instruction-datasets", "nli", "MedNLI")

dataset = ds.load_from_disk(dataset_path)

def show_in_instruction_mode(index):
    print("### Instruction: \n" + dataset[index]["instruction"] + "\n")
    print("### Input: \n" + dataset[index]["input"] + "\n")
    print("### Output: \n" + dataset[index]["output"] + "\n")

show_in_instruction_mode(30)
print()
