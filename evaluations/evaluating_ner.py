from datasets import load_metric
import seqeval
from load_dataset_chunk import get_dataset
import os
import pickle

groundtruths = []
generated = []
types = []

metric = load_metric("seqeval")

home_path = "[PATH]"
# dataset_path = "[PATH]/Clinical-LLM/clinical-LLM/instruction-datasets/ner/ner-instruction-eval/"
dataset_path = os.path.join(home_path, "results", "ner")
load_path = os.path.join(dataset_path, "NCBI-llama")

# datasetTag = "i2b2-2012"
generated, groundtruths = get_dataset(os.path.join(load_path, "generated-7-base.pickle"), os.path.join(load_path, "groundtruth-7-base.pickle"), None, None)

unformatted_groundtruth = groundtruths
unformatted_generated = generated

# print(generated[2])

print(len(groundtruths))
print(len(generated))

def convert_to_ner_format(entities):
    outputs = []
    for ground in entities:
        items = {}
        # items = []
        for entity in ground.strip().split("\n"):
            try:
                items[entity.split(":")[0].strip()] = entity.split(":")[1].strip()
                # items.append(entity.split(":")[1].strip())
            except:
                pass
        outputs.append(items)

    return outputs

def format_generated(generated):
    final_generated = []
    for gen in generated:
        gen = gen.strip()
        gen = gen.split("### Output:")[2].strip()
        gen = gen.split("###")[0].strip()
        final_generated.append(gen)
    return final_generated

generated = convert_to_ner_format(format_generated(generated))
groundtruths = convert_to_ner_format(groundtruths)

print(len(groundtruths))
print(len(generated))

allLabels = []
count = 0

final_generated = []
final_groundtruth = []

for index, (gen, ground) in enumerate(zip(generated, groundtruths)):
    finalGen = []
    finalGround = []
    
    count += 1
    
    for ground, type in ground.items():
        if type in ["O", "B", "I"]:
            finalGround.append(type)
        else:
            finalGround.append("O")
        
        try:
            if gen[ground] in ["O", "B", "I"]:
                finalGen.append(gen[ground])
            else:
                finalGen.append("O")
        except:
            finalGen.append("O")
    
    final_generated.append(finalGen)
    final_groundtruth.append(finalGround)
    
    
    # print(finalGen)
    # print(finalGround)
    # print()


print(set(allLabels))

print("Final Count = " + str(len(final_groundtruth)))

def compute_metrics(predictions, labels):

    results = metric.compute(
        predictions=predictions, references=labels)

    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }

    return flattened_results

print(compute_metrics(final_generated, final_groundtruth))