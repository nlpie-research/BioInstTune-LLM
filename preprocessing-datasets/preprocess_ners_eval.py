import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"

ner_path = os.path.join(current_path, "instruction-datasets", "ner")
save_path = os.path.join(current_path, "instruction-datasets", "ner", "ner-instruction-eval")

# ncbi = ds.load_from_disk(os.path.join(ner_path, "NCBI-disease-eval"))
# ncbi = ncbi.add_column("type", ["NCBI-disease"] * len(ncbi))

# bc5cdr_disease = ds.load_from_disk(os.path.join(ner_path, "BC5CDR-disease-eval"))
# bc5cdr_disease = bc5cdr_disease.add_column("type", ["BC5CDR-disease"] * len(bc5cdr_disease))

# bc5cdr_chem = ds.load_from_disk(os.path.join(ner_path, "BC5CDR-chem-eval"))
# bc5cdr_chem = bc5cdr_chem.add_column("type", ["BC5CDR-chem"] * len(bc5cdr_chem))

# bc2gm = ds.load_from_disk(os.path.join(ner_path, "BC2GM-eval"))
# bc2gm = bc2gm.add_column("type", ["BC2GM"] * len(bc2gm))

# jnlpba = ds.load_from_disk(os.path.join(ner_path, "JNLPBA-eval"))
# jnlpba = jnlpba.add_column("type", ["JNLPBA"] * len(jnlpba))


# datasets = [
#     ncbi,
#     bc5cdr_disease,
#     bc5cdr_chem,
#     bc2gm,
#     jnlpba
# ]

# dataset = ds.concatenate_datasets(datasets)
# print(dataset)

dataset = ds.load_from_disk(save_path)

def show_in_instruction_mode(index):
    print("### Instruction: \n" + dataset[index]["instruction"] + "\n")
    print("### Input: \n" + dataset[index]["input"] + "\n")
    print("### Output: \n" + dataset[index]["output"] + "\n")


print(dataset)

show_in_instruction_mode(1)
print()
show_in_instruction_mode(2)

# dataset.save_to_disk(save_path)