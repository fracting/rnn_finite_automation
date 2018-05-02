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
    raw_dataset = []
    f = open(path)
    file = f.read()
    for line in file.split("\n"):
        (x,y) = line.split(",")
        xs = list(x)
        raw_dataset.append((xs, y))
    f.close()
    return raw_dataset

def load_dataset(path, cont_train_size, rand_train_size, continuous_validation_size, random_validation_size):
    print("raw dataset: %s" % path)
    raw_dataset = load_raw_data(path)
    size = len(raw_dataset)
    print("first 10 rows of raw dataset:")
    for i in range(10):
        print(raw_dataset[i])
    print("")
    print("continuous training size: %d" % cont_train_size)
    print("continuous validation size: %d" % continuous_validation_size)
    print("random training size: %d" % rand_train_size)
    print("random validation size: %d" % random_validation_size)

    dataset = {}
    dataset['cont_train'] = raw_dataset[0:cont_train_size]
    dataset['cont_valid'] = raw_dataset[cont_train_size:cont_train_size+continuous_validation_size]

    random_part = raw_dataset[cont_train_size+continuous_validation_size:]
    random_indices = torch.randperm(len(random_part))
    random_indices = [entry.item() for entry in random_indices]
    random_part = [random_part[index] for index in random_indices]

    dataset['rand_train'] = random_part[0:rand_train_size]
    dataset['rand_valid'] = random_part[rand_train_size:rand_train_size+random_validation_size]

    for data_name in dataset:
        print("first 10 rows of %s" % data_name)
        for i in range(10):
            print(dataset[data_name][i])
        print("")

    return dataset

char_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
category_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
