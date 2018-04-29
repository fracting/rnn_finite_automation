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

training_data = [
  (list("00000"), "1"),
  (list("12345"), "0"),
  (list("56789"), "0"),
  (list("12348"), "1")
]

char_to_ix = {}
for seq, label in training_data:
    for char in seq:
        if char not in char_to_ix:
            char_to_ix[char] = len(char_to_ix)

category_to_ix = {"0": 0, "1": 1}
