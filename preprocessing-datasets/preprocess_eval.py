import datasets as ds
import numpy as np

import os
import random

# current_path = "/Users/mohammadmahdi/Desktop/__________My_Future/_Clinical Models/Clinical-LLM/clinical-LLM/"
current_path = "[PATH]/Clinical-LLM/clinical-LLM/"

path = os.path.join(current_path, "instruction-datasets")

i2b2_2012 = ds.load_from_disk(os.path.join(path, "ner", "i2b2-2012-eval"))
i2b2_2012 = i2b2_2012.add_column("type", ["i2b2-2012"] * len(i2b2_2012))

i2b2_2010 = ds.load_from_disk(os.path.join(path, "re", "i2b2-2010-eval"))
i2b2_2010 = i2b2_2010.add_column("type", ["i2b2-2010"] * len(i2b2_2010))

medNLI = ds.load_from_disk(os.path.join(path, "nli", "MedNLI-eval"))
medNLI = medNLI.add_column("type", ["MedNLI"] * len(medNLI))

hoc = ds.load_from_disk(os.path.join(path, "cls", "hoc-eval"))
hoc = hoc.add_column("type", ["hoc"] * len(hoc))

datasets = [
    i2b2_2012,
    i2b2_2010,
    medNLI,
    hoc
]

dataset = ds.concatenate_datasets(datasets)
print(dataset)

def show_in_instruction_mode(index):
    print("### Instruction: \n" + dataset[index]["instruction"] + "\n")
    print("### Input: \n" + dataset[index]["input"] + "\n")
    print("### Output: \n" + dataset[index]["output"] + "\n")

show_in_instruction_mode(10)
print()
show_in_instruction_mode(100)

dataset.save_to_disk(os.path.join(path, "eval"))
