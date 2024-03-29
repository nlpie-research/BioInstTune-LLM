import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"

path = os.path.join(current_path, "clinical-datasets", "MedNLI-eval")
save_path = os.path.join(current_path, "instruction-datasets", "nli", "MedNLI-eval")

dataset = ds.load_from_disk(path)
index_to_label = {0: 'Contradiction', 1: 'Neutral', 2: 'Entailment'}

instructions = [
    "Your goal is to determine the relationship between the two provided clinical sentences and classify them into one of the following categories:\nContradiction: If the two sentences contradict each other.\nNeutral: If the two sentences are unrelated to each other.\nEntailment: If one of the sentences logically entails the other.",
    "Your task is to assess the connection between the given clinical sentences and place them into one of these categories:\nContradiction: If the two sentences conflict or are in opposition.\nNeutral: If there is no clear logical connection between the sentences.\nEntailment: If one sentence can be logically inferred or implied by the other.",
    "Your mission is to identify the logical relationship between the two clinical sentences and categorize them as:\nContradiction: If the sentences contradict each other in their meaning.\nNeutral: If there is no significant connection or logical inference between the sentences.\nEntailment: If one sentence logically implies or entails the other.",
    "In the provided clinical sentences, your objective is to determine their relationship and assign one of the following labels:\nContradiction: If the sentences present conflicting information.\nNeutral: If there is no apparent logical relationship between the sentences.\nEntailment: If one sentence logically implies or necessitates the other.",
    "Your task involves evaluating the connection between the two clinical sentences and classifying them into one of these categories:\nContradiction: If the sentences are in direct opposition or conflict.\nNeutral: If there is no clear logical association between the sentences.\nEntailment: If one sentence logically follows or implies the other.",
]

def mapping_function(batch):
    dataDict = {
        "instruction": [],
        "input": [],
        "output": [],
    }
    for sentece_1, sentence_2, label in zip(batch["sentence1"], batch["sentence2"], batch["labels"]):
        instruction = random.choice(instructions)
        text = f"Sentence 1: {sentece_1.strip()}\nSentence 2: {sentence_2.strip()}"
        output = index_to_label[label]

        dataDict["instruction"].append(instruction.strip())
        dataDict["input"].append(text.strip())
        dataDict["output"].append(output.strip())

    return dataDict

mapped_dataset = dataset.map(mapping_function, batched=True, remove_columns=dataset.column_names)

print(mapped_dataset)

def show_in_instruction_mode(index):
    print("### Instruction: \n" + mapped_dataset[index]["instruction"] + "\n")
    print("### Input: \n" + mapped_dataset[index]["input"] + "\n")
    print("### Output: \n" + mapped_dataset[index]["output"] + "\n")

show_in_instruction_mode(200)
print()
show_in_instruction_mode(300)

# mapped_dataset.save_to_disk(save_path)