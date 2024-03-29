import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"

ner_path = os.path.join(current_path, "instruction-datasets", "ner")
save_path = os.path.join(current_path, "instruction-datasets", "ner", "ner-instruction-with-source")

source = ["NCBI-disease", "BC5CDR-disease", "BC5CDR-chem", "BC2GM", "JNLPBA", "i2b2-2012"]

datasets = [
    ds.load_from_disk(os.path.join(ner_path, "NCBI-disease")),
    ds.load_from_disk(os.path.join(ner_path, "BC5CDR-disease")),
    ds.load_from_disk(os.path.join(ner_path, "BC5CDR-chem")),
    ds.load_from_disk(os.path.join(ner_path, "BC2GM")),
    ds.load_from_disk(os.path.join(ner_path, "JNLPBA")),
    ds.load_from_disk(os.path.join(ner_path, "i2b2-2012"))
]

for index, dataset in enumerate(datasets):
  datasets[index] = datasets[index].add_column("source", source[index] * len(dataset))

dataset = ds.concatenate_datasets(datasets)
print(dataset)

dataset = dataset.shuffle()

# dataset = ds.load_from_disk(save_path)

print(dataset)

def show_in_instruction_mode(index):
    print("### Instruction: \n" + dataset[index]["instruction"] + "\n")
    print("### Input: \n" + dataset[index]["input"] + "\n")
    print("### Output: \n" + dataset[index]["output"] + "\n")

show_in_instruction_mode(5)
print()
show_in_instruction_mode(10)

# dataset.save_to_disk(save_path)
