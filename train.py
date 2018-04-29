import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import DFA
from data import char_to_ix, category_to_ix, seqs_to_tensor, categories_to_tensor, training_data

EMBEDDING_DIM = 6
HIDDEN_DIM = 5
NUM_LAYERS = 2
BATCH_SIZE = 4

print_per_epoch = 100

torch.manual_seed(1) # TODO - disable manual seed in production version

model = DFA(EMBEDDING_DIM, HIDDEN_DIM, len(char_to_ix), len(category_to_ix), NUM_LAYERS, BATCH_SIZE)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    seqs, _ = list(zip(*training_data))
    inputs = seqs_to_tensor(seqs, char_to_ix)
    category_scores = model(inputs)
    print("category scores before training: \n" + str(category_scores))

for epoch in range(1000):
    model.zero_grad()
    model.hidden = model.init_hidden()

    training_size = len(training_data)
    permutation = np.random.permutation(training_size)

    for i in range(0, training_size, BATCH_SIZE):
    
        indices = list(permutation[i:i+BATCH_SIZE])
        batch_data = [training_data[index] for index in indices]

        seqs, categories = list(zip(*batch_data))
        seqs = list(seqs)
        categories = list(categories)
        seqs_in = seqs_to_tensor(seqs, char_to_ix)
        targets = categories_to_tensor(categories, category_to_ix)

        category_scores = model(seqs_in)

        loss = loss_function(category_scores, targets)
        if epoch % print_per_epoch == 0:
            print("epoch %d loss %f" % (epoch, loss.data))

        loss.backward()
        optimizer.step()

with torch.no_grad():
    seqs, _ = list(zip(*training_data))
    inputs = seqs_to_tensor(seqs, char_to_ix)
    category_scores = model(inputs)
    print("category scores after training: \n" + str(category_scores))
