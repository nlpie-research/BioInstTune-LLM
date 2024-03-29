import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import datasets as ds
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers as ts
import pickle
from accelerate import Accelerator

generated_outputs = []
groundtruth_outptus = []
#types = []

model_path = "[PATH]/Clinical-LLM/clinical-LLM/Llama-2-7b-chat-hf/"
path = "[PATH]/Clinical-LLM/clinical-LLM/instruction-datasets/"

dataset_path = "[PATH]/Clinical-LLM/clinical-LLM/instruction-datasets/nli/MedNLI-eval"
save_path = "[PATH]/Clinical-LLM/clinical-LLM/instruction-datasets/nli/MedNLI-eval/outputs/"

#dataset_path = os.path.join(path, "re", "i2b2-2010-eval")
#save_path = os.path.join(dataset_path, "outputs")

output_path = os.path.join(save_path, "output-base.txt")
progress_path = os.path.join(save_path, "progress-base.txt")

with open(output_path, mode="w") as f:
    f.write("")

with open(progress_path, mode="w") as f:
    f.write("")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.unk_token

# model = AutoModelForCausalLM.from_pretrained("chaoyi-wu/PMC_LLAMA_7B", device_map="balanced_low_0", torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced", torch_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained(os.path.join(model_path, "checkpoints", "checkpoint-13000"), device_map="balanced", torch_dtype=torch.float16)

# model.to_bettertransformer()

dataset = ds.load_from_disk(dataset_path)

print(dataset)

for item in dataset:
    groundtruth_outptus.append(item["output"])
#    types.append(item["type"])

#dataset = dataset.remove_columns(["type"])

def mapping_function(dataset):
    dataDict = {
        "text": [],
    }

    for instruction, input in zip(dataset["instruction"], dataset["input"]):
        text = f'### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n'
        dataDict["text"].append(text)

    tokenizer_output = tokenizer(dataDict["text"])

    return tokenizer_output

tokenized_dataset = dataset.map(mapping_function, batched=True, remove_columns=["instruction", "input", "output"])

print(tokenized_dataset)

data_collator = ts.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

def collator_function(dataset):
    dataDict = {
        key: [] for key in dataset[0].keys()
    }

    for item in dataset:
        for key in item.keys():
            dataDict[key].append(item[key])

    outputs = data_collator(dataDict)

    for key in outputs.keys():
        outputs[key] = outputs[key].to(Accelerator().process_index)

    return outputs

data_loader = DataLoader(tokenized_dataset, shuffle=False, batch_size=1, collate_fn=collator_function)

progress_bar = tqdm(range(len(data_loader)))

for index, data in enumerate(data_loader):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        outputs = model.generate(**data, max_new_tokens=512, num_beams=4)

    outputs[outputs == -100] = tokenizer.pad_token_id

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    with open(output_path, mode="a") as f:
        for decoded_output in list(decoded_outputs):
            f.write(decoded_output + "\n\n")

    with open(progress_path, mode="w") as f:
        f.write("current step: " + str(index) + " / " + str(len(data_loader)))

    generated_outputs += list(decoded_outputs)

    progress_bar.update(1)

with open(os.path.join(save_path, "generated-7-base.pickle"), "wb") as f:   #Pickling
    pickle.dump(generated_outputs, f)

with open(os.path.join(save_path, "groundtruth-7-base.pickle"), "wb") as f:
    pickle.dump(groundtruth_outptus, f)

#with open(os.path.join(save_path, "types-7-base.pickle"), "wb") as f:
#     pickle.dump(types, f)
