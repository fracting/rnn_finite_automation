import torch

def seqs_to_tensor(seqs, to_ix):
    # TODO: padding
    idxs = [[to_ix[w] for w in seq] for seq in seqs]
    # transpose to shape(len, batch_size)
    idxs_tensor = torch.tensor(idxs, dtype=torch.long).transpose(0,1)
    return idxs_tensor

def categories_to_tensor(categories, to_ix):
    idxs = [to_ix[w] for w in categories]
    idxs_tensor = torch.tensor(idxs, dtype=torch.long)
    return idxs_tensor

def load_raw_data(path):
    dataset = []
    f = open(path)
    file = f.read()
    for line in file.split("\n"):
        (x,y) = line.split(",")
        xs = list(x)
        dataset.append((xs, y))
    f.close()
    return dataset

def load_dataset(path, continuous_training_size, random_training_size, continuous_validation_size, random_validation_size):
    print("dataset: %s" % path)
    dataset = load_raw_data(path)
    size = len(dataset)
    print("first 10 rows of dataset:")
    for i in range(10):
        print(dataset[i])
    print("")
    print("continuous training size: %d" % continuous_training_size)
    print("continuous validation size: %d" % continuous_validation_size)
    print("random training size: %d" % random_training_size)
    print("random validation size: %d" % random_validation_size)

    continuous_training_data = dataset[0:continuous_training_size]
    continuous_validation_data = dataset[continuous_training_size:continuous_training_size+continuous_validation_size]

    random_part = dataset[continuous_training_size+continuous_validation_size:]
    random_indices = torch.randperm(len(random_part))
    random_indices = [entry.item() for entry in random_indices]
    random_part = [random_part[index] for index in random_indices]

    random_training_data = random_part[0:random_training_size]
    random_validation_data = random_part[random_training_size:random_training_size+random_validation_size]

    return continuous_training_data, random_training_data, continuous_validation_data, random_validation_data

char_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
category_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
