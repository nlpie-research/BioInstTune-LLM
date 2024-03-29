import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import datasets as ds
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers as ts
import pickle
from accelerate import Accelerator
# local_rank = int(os.getenv("LOCAL_RANK", "0"))
# world_size = int(os.getenv("WORLD_SIZE", "1"))
# torch.cuda.set_device(local_rank)
generated_outputs = []
groundtruth_outptus = []
# types = []

model_path = "[PATH]/Clinical-LLM/clinical-LLM/Llama-2-13b-chat-hf/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.unk_token

model = AutoModelForCausalLM.from_pretrained(os.path.join(model_path, "checkpoints", "checkpoint-13000"), device_map="balanced", torch_dtype=torch.float16)
# model = AutoModelForCausalLM.from_pretrained(os.path.join(model_path), device_map="balanced", torch_dtype=torch.float16)

model.generation_config.pad_token_id = tokenizer.pad_token_id

input_text = """
### Instruction:
In the clinical text, your goal is to determine connections between medical problems, treatments, and tests. The markers @problem$, @test$, and @treatment$ are used to tag these entities in the text. Categorize the relation between two entities as one of the following options:
Treatment improves medical problem (TrIP)
Treatment worsens medical problem (TrWP)
Treatment causes medical problem (TrCP)
Treatment is administered for medical problem (TrAP)
Treatment is not administered because of medical problem (TrNAP)
Test reveals medical problem (TeRP)
Test conducted to investigate medical problem (TeCP)
Medical problem indicates medical problem (PIP)
No Relations

### Input:
@problem$ demonstrated @test$ .

### Output:

"""

inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs, max_new_tokens=512)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])