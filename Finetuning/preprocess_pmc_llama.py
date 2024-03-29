import os
import json

import datasets as ds

import random

dataset = ds.load_dataset("axiong/pmc_llama_instructions")["train"]
dataset = dataset.remove_columns("sample_id")

umls_relation = dataset.filter(lambda x: x["source"] == "umls_relation").remove_columns("source")
umls = dataset.filter(lambda x: x["source"] == "umls").remove_columns("source")
pubmedQA = dataset.filter(lambda x: x["source"] == "pubmedqa.ori_pqaa").remove_columns("source")
medQA = dataset.filter(lambda x: x["source"] == "medmcqa").remove_columns("source")

pubmedQA = pubmedQA.shuffle(len(pubmedQA)).select(range(50000))
medQA = medQA.shuffle(len(medQA)).select(range(50000))

pmc_llama = ds.concatenate_datasets((umls_relation, umls, pubmedQA, medQA))
pmc_llama = pmc_llama.shuffle(len(pmc_llama))

print(pmc_llama)

pmc_llama.save_to_disk(os.path.join("[PATH]/Clinical-LLM/clinical-LLM", "instruction-datasets", "pmc-llama-v2"))