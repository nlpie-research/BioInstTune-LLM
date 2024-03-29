import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"
dataset_path = os.path.join(current_path, "clinical-datasets", "i2b2-2010-re-eval")
save_path = os.path.join(current_path, "instruction-datasets", "re", "i2b2-2010-eval")

dataset = ds.load_from_disk(dataset_path)

index_to_label = {
    0: 'PIP',
    1: 'TeCP',
    2: 'TeRP',
    3: 'TrAP',
    4: 'TrCP',
    5: 'TrIP',
    6: 'TrNAP',
    7: 'TrWP',
    8: 'No Relations'
}

instructions = [
    "Your task is to determine the relationships between medical problems, treatments, and tests within the clinical text. Medical problems are marked as @problem$, medical tests are marked as @test$, and treatments are marked as @treatment$. Categorize the relationship between two entities in the text as one of the following options:\nTreatment improves medical problem (TrIP)\nTreatment worsens medical problem (TrWP)\nTreatment causes medical problem (TrCP)\nTreatment is administered for medical problem (TrAP)\nTreatment is not administered because of medical problem (TrNAP)\nTest reveals medical problem (TeRP)\nTest conducted to investigate medical problem (TeCP)\nMedical problem indicates medical problem (PIP)\nNo Relations",
    "In the clinical text, your objective is to identify relationships between medical problems, treatments, and tests. Medical problems are tagged as @problem$, medical tests as @test$, and treatments as @treatment$. Classify the relationship between two entities as one of the following:\nTreatment improves medical problem (TrIP)\nTreatment worsens medical problem (TrWP)\nTreatment causes medical problem (TrCP)\nTreatment is administered for medical problem (TrAP)\nTreatment is not administered because of medical problem (TrNAP)\nTest reveals medical problem (TeRP)\nTest conducted to investigate medical problem (TeCP)\nMedical problem indicates medical problem (PIP)\nNo Relations",
    "In the clinical text, your goal is to determine connections between medical problems, treatments, and tests. The markers @problem$, @test$, and @treatment$ are used to tag these entities in the text. Categorize the relation between two entities as one of the following options:\nTreatment improves medical problem (TrIP)\nTreatment worsens medical problem (TrWP)\nTreatment causes medical problem (TrCP)\nTreatment is administered for medical problem (TrAP)\nTreatment is not administered because of medical problem (TrNAP)\nTest reveals medical problem (TeRP)\nTest conducted to investigate medical problem (TeCP)\nMedical problem indicates medical problem (PIP)\nNo Relations",
]

def mapping_function(batch):
    dataDict = {
        "instruction": [],
        "input": [],
        "output": [],
    }

    for text, label in zip(batch["sentence"], batch["labels"]):
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

show_in_instruction_mode(101)
print()

# mapped_dataset.save_to_disk(save_path)