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

def show_dataset(dataset, rows):
    if rows > 0:
        print("\n".join([str(row) for row in dataset[:rows]]))
    else:
        print("\n".join([str(row) for row in dataset[rows:]]))
    print("\n")

def load_dataset(path, cont_train_size, rand_train_size, cont_valid_size, rand_valid_size):
    print("raw dataset: %s" % path)
    raw_dataset = load_raw_data(path)
    size = len(raw_dataset)
    print("size of raw dataset: %d" % size)
    assert cont_train_size + rand_train_size + cont_valid_size + rand_valid_size < size
    print("first 10 rows of raw dataset:")
    show_dataset(raw_dataset, 10)
    print("last 10 rows of raw dataset:")
    show_dataset(raw_dataset, -10)

    print("continuous training size: %d" % cont_train_size)
    print("continuous validation size: %d" % cont_valid_size)
    print("random training size: %d" % rand_train_size)
    print("random validation size: %d" % rand_valid_size)
    print("")

    dataset = {}
    dataset['cont_train'] = raw_dataset[0:cont_train_size]
    dataset['cont_valid'] = raw_dataset[-cont_valid_size:]

    random_part = raw_dataset[cont_train_size:-cont_valid_size]
    random_indices = torch.randperm(len(random_part)).tolist()
    random_part = [random_part[index] for index in random_indices]

    dataset['rand_train'] = random_part[0:rand_train_size]
    dataset['rand_valid'] = random_part[rand_train_size:rand_train_size+rand_valid_size]

    for data_name in dataset:
        print("first 10 rows of %s" % data_name)
        show_dataset(dataset[data_name], 10)

    seqs, categories = list(zip(*raw_dataset))
    categories = set(categories)
    category_size = len(categories)
    seqs = list(seqs)
    vocabs = [vocab for seq in seqs for vocab in seq]
    vocab_size = len(set(vocabs))
    return dataset, vocab_size, category_size

char_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
category_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15}
