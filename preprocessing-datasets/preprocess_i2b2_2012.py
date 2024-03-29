import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"

path = os.path.join(current_path, "clinical-datasets", "ner", "i2b2-2012-eval")
save_path = os.path.join(current_path, "instruction-datasets", "ner", "i2b2-2012-eval")

dataset = ds.load_from_disk(path)
# index_to_label = {0: 'O', 1: 'I', 2: 'B'}

instructions = [
    "Your task is to identify clinical Named Entities within the text and apply the BIO labeling scheme. Use the following labels to categorize each entity:\nOCCURRENCE: If the entity represents a clinical incident or event.\nPROBLEM: If the entity indicates a medical problem.\nTEST: If the entity pertains to a medical test.\nTREATMENT: If the entity refers to a medical treatment.\nEVIDENTIAL: If the entity provides evidence.\nCLINICAL_DEPT: If the entity relates to a clinical department.\nO: If the entity doesn't fit into any of the above categories.",
    "In your role, you are tasked with detecting clinical Named Entities within the text. Implement the BIO labeling scheme and use the following labels to classify each entity:\nOCCURRENCE: If the entity signifies a clinical incident or event.\nPROBLEM: If the entity denotes a medical problem.\nTEST: If the entity represents a medical test.\nTREATMENT: If the entity corresponds to a medical treatment.\nEVIDENTIAL: If the entity offers evidence.\nCLINICAL_DEPT: If the entity relates to a clinical department.\nO: If the entity does not fall into any of the above categories.",
    "Your goal as an annotator is to recognize clinical Named Entities in the text and apply the BIO labeling scheme. Use the following labels to classify each entity:\nOCCURRENCE: If the entity describes a clinical incident or event.\nPROBLEM: If the entity denotes a medical problem.\nTEST: If the entity represents a medical test.\nTREATMENT: If the entity refers to a medical treatment.\nEVIDENTIAL: If the entity provides evidence.\nCLINICAL_DEPT: If the entity relates to a clinical department.\nO: If the entity does not fit into any of the above categories.",
    "In your capacity, you are responsible for detecting clinical Named Entities within the text. Apply the BIO labeling scheme and use the following labels to categorize each entity:\nOCCURRENCE: If the entity signifies a clinical incident or event.\nPROBLEM: If the entity indicates a medical problem.\nTEST: If the entity pertains to a medical test.\nTREATMENT: If the entity refers to a medical treatment.\nEVIDENTIAL: If the entity provides evidence.\nCLINICAL_DEPT: If the entity relates to a clinical department.\nO: If the entity does not fall into any of the above categories.",
    "Your role involves identifying clinical Named Entities in the text and applying the BIO labeling scheme. Utilize the following labels to classify each entity:\nOCCURRENCE: If the entity represents a clinical incident or event.\nPROBLEM: If the entity denotes a medical problem.\nTEST: If the entity corresponds to a medical test.\nTREATMENT: If the entity relates to a medical treatment.\nEVIDENTIAL: If the entity provides evidence.\nCLINICAL_DEPT: If the entity relates to a clinical department.\nO: If the entity does not fit into any of the above categories.",
]

print(dataset)

def generate_output(tokens, labels):
    text = ''
    for token, label in zip(tokens, labels):
        # text += token + " : " + index_to_label[label] + "\n"
        text += token + " : " + label + "\n"
    return text

def mapping_function(batch):
    outputDict = {
        "instruction": [],
        "input": [],
        "output": []
    }

    for tokens, labels in zip(batch["tokens"], batch["ner_tags_str"]):
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

show_in_instruction_mode(10)
print()
show_in_instruction_mode(100)

# mapped_dataset.save_to_disk(save_path)

