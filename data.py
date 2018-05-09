import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def onehot(x, vocab_size):
    x_onehot = torch.zeros([d for d in x.shape] + [vocab_size])
    x = x.view([d for d in x.shape] + [1])
    x_onehot.scatter_(1, x, 1)
    return x_onehot

def seqs_to_tensor(seqs, to_ix, vocab_size, embedding_dim, embedding_model):
    # TODO: padding
    idxs = [[to_ix[w] for w in seq] for seq in seqs]
    # transpose to shape(len, batch_size)
    idxs_tensor = torch.tensor(idxs, dtype=torch.long).transpose(0,1)
    embeddings_seq = []
    for batch_elems in idxs_tensor:
        batch_elems_onehot = onehot(batch_elems, vocab_size)
        _, embeddings = embedding_model(batch_elems_onehot)
        embeddings_seq.append(embeddings)
    return torch.stack(embeddings_seq)

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

class embed(nn.Module):

    def __init__(self, vocab_size, embedding_dim, batch_size):

        super(embed, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.input2embedding = nn.Linear(vocab_size, embedding_dim)
        self.embedding2output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input):

        embedding = self.input2embedding(input)
        embedding = F.tanh(embedding)
        output = self.embedding2output(embedding)
        output = F.log_softmax(output, dim=1)
    
        return output, embedding

def train_embedding(vocab_size, embedding_dim, batch_size):
    input = torch.randint(0, batch_size, (batch_size,)).long()
    input = input % vocab_size
    input_onehot = onehot(input, vocab_size)

    embedding_model = embed(vocab_size, embedding_dim, batch_size)
    loss_function = nn.NLLLoss()
    learning_rate = 0.05

    for epoch in range(0, 100):
        optimizer = optim.Adam(embedding_model.parameters(), lr=learning_rate)
        embedding_model.zero_grad()
        output, embedding = embedding_model(input_onehot)
        loss = loss_function(output, input)
        loss.backward()
        optimizer.step()

    print("final embedding loss: ", loss)

    return embedding_model
