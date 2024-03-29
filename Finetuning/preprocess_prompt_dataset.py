from config import *
import os
import json

import datasets as ds

import random

HEALTH_CARE_MAGIC = os.path.join(current_path, "Datasets", "HealthCareMagic-100k.json")

datas = json.load(open(HEALTH_CARE_MAGIC, mode="r"))

dataDict = {
    "instruction": [],
    "input": [],
    "output": [],
}

instructions = [
    "As a healthcare expert, provide answers to medical inquiries based on the information given by the user.",
    "In your role as a medical professional, address the user's medical questions and concerns.",
    "If you have medical expertise, assist the user by responding to their healthcare-related questions.",
    "Your task is to offer medical advice and answers to questions posed by users regarding their health.",
    "As a virtual doctor, respond to the user's medical queries and provide relevant guidance.",
    "If you possess medical knowledge, assist users by addressing their health-related questions.",
    "In your capacity as a healthcare expert, offer insights and recommendations in response to users' medical inquiries.",
    "Your role involves answering medical questions and offering advice to users based on their descriptions.",
    "As a medical chatbot, your responsibility is to provide information and guidance on medical matters to users.",
    "If you have expertise in healthcare, assist users by addressing their medical questions and concerns."
]

for data in datas:
    dataDict["instruction"].append(random.choice(instructions))
    dataDict["input"].append(data["input"])
    dataDict["output"].append(data["output"])

dataset = ds.Dataset.from_dict(dataDict)
dataset = dataset.shuffle(len(dataset)).select(range(int(len(dataset) * 0.5)))

print(dataset)
print(dataset[0])

dataset.save_to_disk(os.path.join(save_dataset_path, "chatdoctor"))