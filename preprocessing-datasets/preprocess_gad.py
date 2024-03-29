import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"
dataset_path = os.path.join(current_path, "biomedical-datasets", "re", "gad-eval")
save_path = os.path.join(current_path, "instruction-datasets", "re", "gad-eval")

dataset = ds.load_from_disk(dataset_path)

index_to_label = {
    1: 'Negative',
    0: 'Positive',
}

instructions = [
    "In your capacity as a medical expert, your objective is to ascertain the connections between genes and diseases within the clinical text. Genes are denoted as @GENE$, and diseases are indicated as @DISEASE$. Categorize the relationship between genes and diseases in the text as one of the following options:\nPositive: If there is a clear relation between the mentioned gene and disease in the text.\nNegative: If there is no apparent relation between the mentioned gene and disease in the text.",
    "Your role as a medical expert involves identifying the associations between genes and diseases in the clinical text. Genes are labeled as @GENE$, and diseases are marked as @DISEASE$. Classify the relationship between genes and diseases in the text as either:\nPositive: If there is an evident relation between the mentioned gene and disease in the text.\nNegative: If there is no discernible relation between the mentioned gene and disease in the text.",
    "Your task as a medical expert is to uncover the links between genes and diseases in the provided clinical text. Genes are indicated as @GENE$, and diseases are represented as @DISEASE$. Categorize the relationship between genes and diseases in the text as one of the following options:\nPositive: If there is a clear connection between the mentioned gene and disease in the text.\nNegative: If there is no observable connection between the mentioned gene and disease in the text.",
    "As a medical expert, your mission is to identify the relationships between genes and diseases in the clinical text. Genes are tagged as @GENE$, and diseases are noted as @DISEASE$. Classify the relationship between genes and diseases in the text as either:\nPositive: If there is an evident correlation between the mentioned gene and disease in the text.\nNegative: If there is no discernible correlation between the mentioned gene and disease in the text.",
    "In your role as a medical expert, you are tasked with determining the links between genes and diseases in the clinical text. Genes are designated as @GENE$, and diseases are highlighted as @DISEASE$. Categorize the relationship between genes and diseases in the text as one of the following options:\nPositive: If there is a clear connection between the mentioned gene and disease in the text.\nNegative: If there is no apparent connection between the mentioned gene and disease in the text."
]

def mapping_function(batch):
    dataDict = {
        "instruction": [],
        "input": [],
        "output": [],
    }

    for text, label in zip(batch["sentence"], batch["label"]):
        instruction = random.choice(instructions)
        input = text
        output = index_to_label[label]

        dataDict["instruction"].append(instruction)
        dataDict["input"].append(input)
        dataDict["output"].append(output)

    return dataDict

mapped_dataset = dataset.map(mapping_function, remove_columns=dataset.column_names, batched=True)

def show_in_instruction_mode(index):
    print("### Instruction: \n" + mapped_dataset[index]["instruction"] + "\n")
    print("### Input: \n" + mapped_dataset[index]["input"] + "\n")
    print("### Output: \n" + mapped_dataset[index]["output"] + "\n")

show_in_instruction_mode(10)
print()

# mapped_dataset.save_to_disk(save_path)