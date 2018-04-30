import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import sys

from model import DFA
from data import char_to_ix, category_to_ix, seqs_to_tensor, categories_to_tensor, load_dataset

EMBEDDING_DIM = 4
HIDDEN_DIM = 6
NUM_LAYERS = 1
BATCH_SIZE = 128

print_per_epoch = 100
print_per_batch = 100
total_epoch1 = 10000
total_epoch2 = 10000

learning_rate = 0.0015 * BATCH_SIZE

torch.manual_seed(4) # TODO - disable manual seed in production version

model = DFA(EMBEDDING_DIM, HIDDEN_DIM, len(char_to_ix), len(category_to_ix), NUM_LAYERS, BATCH_SIZE)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

continuous_training_size = 2048
random_training_size = 2048
continuous_validation_size = 1024
random_validation_size = 1024
continuous_training_data, random_training_data, continuous_validation_data, random_validation_data = load_dataset("dataset/10div7.txt", continuous_training_size, random_training_size, continuous_validation_size, random_validation_size)

print("batch size: %d" % BATCH_SIZE)
print("embedding dim: %d" % EMBEDDING_DIM)
print("hidden dim: %d" % HIDDEN_DIM)
print("num layers: %d" % NUM_LAYERS)
print("learning rate: %f" % learning_rate)
print("")

def calc_accuracy(score_tensors, target):
    _, index_tensors = score_tensors.max(dim=1)
    correct_prediction = (index_tensors == target)
    accuracy = correct_prediction.sum().item() / len(correct_prediction)
    return accuracy

def validation(validation_set, validation_name):
    with torch.no_grad():
        validation_size = len(validation_set)
        batch_count = validation_size // BATCH_SIZE
        round_to_batch = batch_count * BATCH_SIZE

        validation_loss = 0
        validation_accuracy = 0
        for i in range(0, round_to_batch, BATCH_SIZE):
            model.zero_grad()
            model.hidden = model.init_hidden()

            batch_data = validation_set[i:i+BATCH_SIZE]
            seqs, categories = list(zip(*batch_data))
            seqs = list(seqs)
            categories = list(categories)
            seqs_in = seqs_to_tensor(seqs, char_to_ix)
            targets = categories_to_tensor(categories, category_to_ix)

            category_scores = model(seqs_in)
            batch_loss = loss_function(category_scores, targets)
            batch_accuracy = calc_accuracy(category_scores, targets)
            validation_loss = validation_loss + batch_loss.data
            validation_accuracy = validation_accuracy + batch_accuracy

        average_loss = validation_loss / batch_count
        average_accuracy = validation_accuracy / batch_count
        print("%s validation loss %f accuracy %f" % (validation_name, average_loss, average_accuracy))
        sys.stdout.flush()

    return average_loss

def train(training_set, training_name, total_epoch):
    print("train %s size %d for %d epoch\n" % (training_name, len(training_set), total_epoch))

    for epoch in range(total_epoch):

        training_size = len(training_set)
        batch_count = training_size // BATCH_SIZE
        round_to_batch = batch_count * BATCH_SIZE
        permutation = torch.randperm(training_size)[0:round_to_batch]
        permutation = [entry.item() for entry in permutation]

        epoch_loss = 0
        epoch_accuracy = 0
        for i in range(0, round_to_batch, BATCH_SIZE):
            model.zero_grad()
            model.hidden = model.init_hidden()

            indices = list(permutation[i:i+BATCH_SIZE])
            batch_data = [training_set[index] for index in indices]

            seqs, categories = list(zip(*batch_data))
            seqs = list(seqs)
            categories = list(categories)
            seqs_in = seqs_to_tensor(seqs, char_to_ix)
            targets = categories_to_tensor(categories, category_to_ix)

            category_scores = model(seqs_in)

            batch_loss = loss_function(category_scores, targets)
            batch_accuracy = calc_accuracy(category_scores, targets)

            epoch_loss = epoch_loss + batch_loss.data
            epoch_accuracy = epoch_accuracy + batch_accuracy
            batch_loss.backward()
            optimizer.step()
        average_loss = epoch_loss / batch_count
        average_accuracy = epoch_accuracy / batch_count

        if epoch % print_per_epoch == 0:
            t_print = datetime.now()
            if epoch > 1:
                t_diff_per_print = t_print - t_last_print
                print("time spent in %d epoch %s" % (print_per_epoch, str(t_diff_per_print)))
            print("%s training epoch %d loss %f accuracy %f" % (training_name, epoch, average_loss, average_accuracy))
            validation(continuous_training_data, "continuous_training")
            validation(continuous_validation_data, "continuous")
            validation(random_validation_data, "random")
            print("")
            sys.stdout.flush()
            t_last_print = datetime.now()

t_begin = datetime.now()
t_print = None
validation(continuous_training_data, "continuous_training")
validation(random_training_data, "random_training")
validation(continuous_validation_data, "continuous_validation")
validation(random_validation_data, "random_validation")
print("")
train(continuous_training_data, "continuous", total_epoch1)
train(continuous_training_data+random_training_data, "continuous+random", total_epoch2)
t_end = datetime.now()
tdiff_begin_end = t_end - t_begin
print("time spent total: %s" % str(tdiff_begin_end))
