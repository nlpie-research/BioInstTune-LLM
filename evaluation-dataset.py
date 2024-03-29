import datasets as ds
import os

datasetPath = "[PATH]/Clinical-LLM/clinical-LLM"

paths = [
    os.path.join(datasetPath, "ner", "ner-instruction-eval"),
    os.path.join(datasetPath, "ner", "i2b2-2012-eval"),
    os.path.join(datasetPath, "nli", "MedNLI-eval"),
    os.path.join(datasetPath, "re", "i2b2-2010-eval"),
    os.path.join(datasetPath, "re", "gad-eval"),
    os.path.join(datasetPath, "cls", "hoc-eval"),
]

print(paths[0])

datasets =  [ds.load_from_disk(path) for path in paths]

finalDataset = ds.concatenate_datasets(datasets)

print(finalDataset)