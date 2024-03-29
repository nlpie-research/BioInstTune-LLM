from Finetuning.config import *
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
import json

import transformers as ts
import datasets as ds

from datasets import Dataset, load_dataset, concatenate_datasets

HEALTH_CARE_MAGIC = os.path.join(current_path, "Datasets", "HealthCareMagic-100k.json")

current_prompt_template = prompt_template

def read_json_dataset(path):
    dataDict = {}

    file = open(path, mode="r")
    loaded_data = json.load(file)
    file.close()


    for row in loaded_data:
        for key in row.keys():
            if key in dataDict:
                dataDict[key].append(row[key])
            else:
                dataDict[key] = [row[key]]

    return Dataset.from_dict(dataDict)

dataset = read_json_dataset(HEALTH_CARE_MAGIC)
dataset = dataset.train_test_split(test_size=0.05)

print(dataset)

#TODO change the preprocessing code
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

def preprocess_function(sample):
    # created prompted input
    # inputs = [prompt_template.format(input=item) for item in sample[text_column]]

    inputs = []
    for instruction, input in zip(sample["instruction"], sample["input"]):
        text = current_prompt_template.format(prompt=instruction, input=input)
        inputs.append(text)

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="do_not_pad")

    labels = []
    for output in sample["output"]:
        labels.append(output + tokenizer.eos_token)

    model_outputs = tokenizer(labels, max_length=1024, truncation=True, padding="do_not_pad")

    final_input_ids = []
    final_labels = []
    for inputs, output in zip(model_inputs["input_ids"], model_outputs["input_ids"]):
        final_input = inputs + output[:-1]
        final_label = [-100] * len(inputs) + output[1:]

        final_input_ids.append(final_input)
        final_labels.append(final_label)


    # print(tokenizer.decode(final_input_ids[0]))

    # for token_1, token_2 in zip(final_input_ids[0], final_labels[0]):
    #     if token_2 != -100:
    #         print("Input: " + tokenizer.decode(token_1) + "\tOutput: " + tokenizer.decode(token_2))

    # print()

    dataDict = {
        "input_ids": final_input_ids,
        "labels": final_labels,
    }

    return dataDict

# process dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features))
print(tokenized_dataset)

# sample_input = [tokenized_dataset[0], tokenized_dataset[1]]

# output = collator_function(sample_input)
# print(output)

# save dataset to disk
tokenized_dataset.save_to_disk(save_dataset_path)
