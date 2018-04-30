import torch
import numpy as np

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

def load_dataset(path, to_random):
   dataset = []
   f = open(path)
   file = f.read()
   for line in file.split("\n"):
       (x,y) = line.split(",")
       xs = list(x)
       dataset.append((xs, y))
   f.close()
   size = len(dataset)
   if to_random:
       random_indices = list(np.random.permutation(size))
       dataset = [dataset[index] for index in random_indices]
   return dataset

char_to_ix = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
category_to_ix = {"0": 0, "1": 1}
