import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import sys

from model import DFA
from data import char_to_ix, category_to_ix, seqs_to_tensor, categories_to_tensor, load_dataset

EMBEDDING_DIM = 6
HIDDEN_DIM = 5
NUM_LAYERS = 2
BATCH_SIZE = 128

print_per_epoch = 50
print_per_batch = 100
total_epoch = 5000

learning_rate = 0.0015 * BATCH_SIZE

torch.manual_seed(1) # TODO - disable manual seed in production version

model = DFA(EMBEDDING_DIM, HIDDEN_DIM, len(char_to_ix), len(category_to_ix), NUM_LAYERS, BATCH_SIZE)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

training_size = 8192
validation_size = 1024
to_random = True
dataset = load_dataset("dataset/10div7.txt", to_random)
all_size = len(dataset)
print("dataset size: %d, random shuffle: %s" % (all_size, str(to_random)))
training_data = dataset[0:training_size]
print("training size: %d" % len(training_data))
validation_data = dataset[-1-validation_size:-1]
print("validation size: %d" % len(validation_data))
print("batch size: %d" % BATCH_SIZE)
print("embedding dim: %d" % EMBEDDING_DIM)
print("hidden dim: %d" % HIDDEN_DIM)
print("num layers: %d" % NUM_LAYERS)
print("learning rate: %f" % learning_rate)
print("\n")

def validation():
    with torch.no_grad():
        validation_size = len(validation_data)
        batch_count = validation_size // BATCH_SIZE
        round_to_batch = batch_count * BATCH_SIZE

        validation_loss = 0
        for i in range(0, round_to_batch, BATCH_SIZE):
            model.zero_grad()
            model.hidden = model.init_hidden()

            batch_data = validation_data[i:i+BATCH_SIZE]
            seqs, categories = list(zip(*batch_data))
            seqs = list(seqs)
            categories = list(categories)
            seqs_in = seqs_to_tensor(seqs, char_to_ix)
            targets = categories_to_tensor(categories, category_to_ix)

            category_scores = model(seqs_in)
            batch_loss = loss_function(category_scores, targets)
            validation_loss = validation_loss + batch_loss.data

        average_loss = validation_loss / batch_count
        print("validation loss %f" % average_loss)
        sys.stdout.flush()

    return average_loss

t_begin = datetime.now()
t_print = None
for epoch in range(total_epoch):

    training_size = len(training_data)
    batch_count = training_size // BATCH_SIZE
    round_to_batch = batch_count * BATCH_SIZE
    permutation = torch.randperm(training_size)[0:round_to_batch]
    permutation = [entry.item() for entry in permutation]

    epoch_loss = 0
    for i in range(0, round_to_batch, BATCH_SIZE):
        model.zero_grad()
        model.hidden = model.init_hidden()
   
        indices = list(permutation[i:i+BATCH_SIZE])
        batch_data = [training_data[index] for index in indices]

        seqs, categories = list(zip(*batch_data))
        seqs = list(seqs)
        categories = list(categories)
        seqs_in = seqs_to_tensor(seqs, char_to_ix)
        targets = categories_to_tensor(categories, category_to_ix)

        category_scores = model(seqs_in)

        batch_loss = loss_function(category_scores, targets)
        #if i // BATCH_SIZE % print_per_batch == 0:
        #    print("batch %d loss %f" % (i // BATCH_SIZE, batch_loss))

        epoch_loss = epoch_loss + batch_loss.data
        batch_loss.backward()
        optimizer.step()
    average_loss = epoch_loss / batch_count

    if epoch % print_per_epoch == 0:
        t_print = datetime.now()
        if epoch > 1:
            t_diff_per_print = t_print - t_last_print
            print("time spent in %d epoch %s" % (print_per_epoch, str(t_diff_per_print)))
        print("epoch %d loss %f" % (epoch, average_loss))
        sys.stdout.flush()
        validation()
        t_last_print = datetime.now()

t_end = datetime.now()
tdiff_begin_end = t_end - t_begin
print("time spent total: %s" % str(tdiff_begin_end))
