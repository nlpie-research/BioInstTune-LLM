import datasets as ds
import numpy as np

import os
import random

current_path = "[PATH]/Clinical-LLM/clinical-LLM/"
dataset_path = os.path.join(current_path, "clinical-datasets", "cls", "hoc-eval")
save_path = os.path.join(current_path, "instruction-datasets", "cls", "hoc-eval")

dataset = ds.load_from_disk(dataset_path)

print(dataset[0])

index_to_label = {
    'sustaining proliferative signaling': 'Sustaining proliferative signaling (PS)',
    'evading growth suppressors': 'Evading growth suppressors (GS)',
    'resisting cell death': 'Resisting cell death (CD)',
    'enabling replicative immortality': 'Enabling replicative immortality (RI)',
    'inducing angiogenesis': 'Inducing angiogenesis (A)',
    'activating invasion and metastasis': 'Activating invasion & metastasis (IM)',
    'genomic instability and mutation': 'Genome instability & mutation (GI)',
    'tumor promoting inflammation': 'Tumor-promoting inflammation (TPI)',
    'cellular energetics': 'Deregulating cellular energetics (CE)',
    'avoiding immune destruction': 'Avoiding immune destruction (ID)',
    'none': 'None',
}

instructions = [
    "Your role as a medical expert involves annotating a provided clinical text to identify the presence of specific cancer-related hallmarks. This task is a multi-class classification, and you are required to assign one or more labels from the following list to the input text if they are applicable:\nSustaining proliferative signaling (PS)\nEvading growth suppressors (GS)\nResisting cell death (CD)\nEnabling replicative immortality (RI)\nInducing angiogenesis (A)\nActivating invasion & metastasis (IM)\nGenome instability & mutation (GI)\nTumor-promoting inflammation (TPI)\nDeregulating cellular energetics (CE)\nAvoiding immune destruction (ID)\nNone",
    "In your capacity as a medical expert, your task is to assess a provided clinical text for indications of specific cancer hallmarks. This is a multi-class classification challenge, and you should assign one or more labels from the list below to the text if they are applicable:\nSustaining proliferative signaling (PS)\nEvading growth suppressors (GS)\nResisting cell death (CD)\nEnabling replicative immortality (RI)\nInducing angiogenesis (A)\nActivating invasion & metastasis (IM)\nGenome instability & mutation (GI)\nTumor-promoting inflammation (TPI)\nDeregulating cellular energetics (CE)\nAvoiding immune destruction (ID)\nNone",
    "Your responsibility as a medical expert is to review a given clinical text for the presence of specific cancer hallmarks. This is a multi-class classification task, and you should assign one or more labels from the provided list to the text if they are applicable:\nSustaining proliferative signaling (PS)\nEvading growth suppressors (GS)\nResisting cell death (CD)\nEnabling replicative immortality (RI)\nInducing angiogenesis (A)\nActivating invasion & metastasis (IM)\nGenome instability & mutation (GI)\nTumor-promoting inflammation (TPI)\nDeregulating cellular energetics (CE)\nAvoiding immune destruction (ID)\nNone",
    "As a medical expert, your task is to analyze a given clinical text and identify specific cancer hallmarks. This task is a multi-class classification, and you are required to assign one or more labels from the following list to the text if they are relevant:\nSustaining proliferative signaling (PS)\nEvading growth suppressors (GS)\nResisting cell death (CD)\nEnabling replicative immortality (RI)\nInducing angiogenesis (A)\nActivating invasion & metastasis (IM)\nGenome instability & mutation (GI)\nTumor-promoting inflammation (TPI)\nDeregulating cellular energetics (CE)\nAvoiding immune destruction (ID)\nNone",
    "In your role as a medical expert, you are tasked with examining a provided clinical text to identify specific cancer hallmarks. This is a multi-class classification assignment, and you should assign one or more labels from the list below to the text if they are relevant:\nSustaining proliferative signaling (PS)\nEvading growth suppressors (GS)\nResisting cell death (CD)\nEnabling replicative immortality (RI)\nInducing angiogenesis (A)\nActivating invasion & metastasis (IM)\nGenome instability & mutation (GI)\nTumor-promoting inflammation (TPI)\nDeregulating cellular energetics (CE)\nAvoiding immune destruction (ID)\nNone",
]

def mapping_function(batch):
    dataDict = {
        "instruction": [],
        "input": [],
        "output": [],
    }

    for text, labels in zip(batch["text"], batch["labels"]):
        instruction = random.choice(instructions)
        input = text
        output = ""

        for label in labels:
            output += index_to_label[label] + ", "

        output = output[:-2]
        dataDict["instruction"].append(instruction)
        dataDict["input"].append(input)
        dataDict["output"].append(output)

    return dataDict

mapped_dataset = dataset.map(mapping_function, remove_columns=dataset.column_names, batched=True)

def show_in_instruction_mode(index):
    print("### Instruction: \n" + mapped_dataset[index]["instruction"] + "\n")
    print("### Input: \n" + mapped_dataset[index]["input"] + "\n")
    print("### Output: \n" + mapped_dataset[index]["output"] + "\n")


for index, item in enumerate(mapped_dataset):
    if item["output"].strip() != "None":
        show_in_instruction_mode(index)
        break

print()
mapped_dataset.save_to_disk(save_path)