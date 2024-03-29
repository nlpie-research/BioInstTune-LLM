import pickle

def get_dataset(generatedPath, groundtruthsPath, typesPath, datasetTag):
    with open(groundtruthsPath, mode="rb") as f:
        groundtruths = pickle.load(f)

    with open(generatedPath, mode="rb") as f:
        generated = pickle.load(f)

    if datasetTag != None:
        with open(typesPath, mode="rb") as f:
            types = pickle.load(f)

    def get_dataset_chunk(types, type):
        startIndex = -1
        endIndex =  -1

        for index, item in enumerate(types):
            if startIndex == -1:
                if item == type:
                    startIndex = index
            else:
                if item != type:
                    endIndex = index
                    break

        if endIndex == -1:
            endIndex = len(types)

        return startIndex, endIndex

    startIndex = 0
    endIndex = len(generated)

    if datasetTag != None:
        startIndex, endIndex = get_dataset_chunk(types, datasetTag)

    print("startIndex: " + str(startIndex))
    print("endIndex: " + str(endIndex))

    groundtruths = groundtruths[startIndex:endIndex]
    generated = generated[startIndex:endIndex]

    return generated, groundtruths
