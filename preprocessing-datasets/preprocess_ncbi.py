import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/clinical-LLM"

path = os.path.join(current_path, "biomedical-datasets", "NCBI-disease-eval")
save_path = os.path.join(current_path, "instruction-datasets", "ner", "NCBI-disease-eval")

dataset = ds.load_from_disk(path)
index_to_label = {0: 'O', 1: 'I', 2: 'B'}

instructions = [
    "Your goal is to detect disease-related Named Entities within the text and apply the BIO labeling scheme. Begin by labeling the first word of a disease-related phrase as B (Begin), and then label the subsequent words in that phrase as I (Inner). Any words not related to diseases should be labeled as O.",
    "In the provided text, your objective is to recognize and label Named Entities associated with diseases using the BIO labeling scheme. Start by marking the beginning of a disease-related phrase with B (Begin), and then continue with I (Inner) for the subsequent words within that phrase. Non-disease words should be labeled as O.",
    "Your task is to spot mentions of diseases in the text and apply the BIO labeling scheme. For each disease-related phrase, label the initial word as B (Begin), and label the rest of the words in the phrase as I (Inner). Any words unrelated to diseases should receive an O label.",
    "In the given text, your mission is to identify Named Entities referring to diseases and employ the BIO labeling scheme. Mark the start of a disease-related phrase with B (Begin), followed by I (Inner) for the remaining words within that phrase. All non-disease terms should be labeled as O.",
    "Your objective is to find instances of diseases within the input text and apply the BIO labeling scheme. Label the first word of each disease-related phrase as B (Begin), and subsequently label the other words in the same phrase as I (Inner). Any words that do not pertain to diseases should be labeled as O.",
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

show_in_instruction_mode(15)
print()
show_in_instruction_mode(30)

# mapped_dataset.save_to_disk(save_path)

