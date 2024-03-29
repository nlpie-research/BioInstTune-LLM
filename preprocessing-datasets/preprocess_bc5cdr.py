import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"

path = os.path.join(current_path, "biomedical-datasets", "BC5CDR-disease-eval")
save_path = os.path.join(current_path, "instruction-datasets", "ner", "BC5CDR-disease-eval")

dataset = ds.load_from_disk(path)
index_to_label = {0: 'O', 1: 'B', 2: 'I'}

instructions = [
    "Your task is to recognize and label Named Entities related to diseases in the text. Use the BIO labeling scheme, marking the first word of a disease-related phrase as B (Begin), and label the subsequent words within that phrase as I (Inner). Words unrelated to diseases should be labeled as O.",
    "In the provided text, your objective is to identify and tag disease-related Named Entities using the BIO labeling scheme. Start by labeling the initial word of a disease-related phrase as B (Begin), and then mark the following words in the same phrase as I (Inner). Any words not pertaining to diseases should receive an O label.",
    "Your task involves annotating the text by identifying and tagging disease-related Named Entities with the BIO labeling scheme. For each disease-related phrase, label the first word as B (Begin), and label the rest of the words within that phrase as I (Inner). Non-disease terms should be labeled as O.",
    "Your mission is to tag disease-related Named Entities in the text using the BIO labeling scheme. When you encounter a disease-related phrase, mark the start with B (Begin) and continue with I (Inner) for the subsequent words in that phrase. Words unrelated to diseases should be labeled as O.",
    "In the provided text, your goal is to identify and label Named Entities associated with diseases. Apply the BIO labeling scheme by designating the first word of a disease-related phrase as B (Begin), and label the remainder of the words within that phrase as I (Inner). Any terms that do not refer to diseases should be labeled as O.",
]


def generate_output(tokens, labels):
    text = ''
    for token, label in zip(tokens, labels):
        text += token + " : " + index_to_label[label] + "\n"
    return text

def mapping_function(batch):
    outputDict = {
        "instruction": [],
        "input": [],
        "output": []
    }

    for tokens, labels in zip(batch["tokens"], batch["ner_tags"]):
        instruction = random.choice(instructions)
        text = ' '.join(tokens)
        output = generate_output(tokens, labels)

        outputDict["instruction"].append(instruction.strip())
        outputDict["input"].append(text.strip())
        outputDict["output"].append(output.strip())

    return outputDict

mapped_dataset = dataset.map(mapping_function, remove_columns=dataset.column_names, batched=True)

print(mapped_dataset)

def show_in_instruction_mode(index):
    print("### Instruction: \n" + mapped_dataset[index]["instruction"] + "\n")
    print("### Input: \n" + mapped_dataset[index]["input"] + "\n")
    print("### Output: \n" + mapped_dataset[index]["output"] + "\n")

show_in_instruction_mode(5)
print()
show_in_instruction_mode(1)

mapped_dataset.save_to_disk(save_path)

