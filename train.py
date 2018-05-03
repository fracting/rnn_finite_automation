import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import sys

from model import DFA
from data import char_to_ix, category_to_ix, seqs_to_tensor, categories_to_tensor, load_dataset

RNN_TYPE = "LSTM"
HIDDEN_DIM = 20
NUM_LAYERS = 1
BATCH_SIZE = 128
DROPOUT = 0.0 # dropout does not apply on output layer, so no effect to single layer network

print_per_epoch = 20
print_per_batch = 100
total_epoch1 = 3000
print("total_epoch1 %d" % total_epoch1)

torch.manual_seed(4) # TODO - disable manual seed in production version

cont_train_size = 8571
rand_train_size = 16384
cont_valid_size = 8571
rand_valid_size = 16384
dataset_path = "10div7.multiclass.txt"
dataset, vocab_size, category_size = load_dataset("dataset/"+dataset_path, cont_train_size, rand_train_size, cont_valid_size, rand_valid_size)
EMBEDDING_DIM = 80

load_model = False
model_path = "checkpoint/10div7.multiclass.60em.20hidden.train.pt"
if load_model:
    print("model_path: " + model_path)
    model = torch.load(model_path)
else:
    model = DFA(RNN_TYPE, EMBEDDING_DIM, HIDDEN_DIM, len(char_to_ix), category_size, NUM_LAYERS, BATCH_SIZE, DROPOUT)
learning_rate = 0.001

loss_function = nn.NLLLoss()

def calc_accuracy(score_tensors, target):
    _, index_tensors = score_tensors.max(dim=1)
    correct_prediction = (index_tensors == target)
    accuracy = correct_prediction.sum().item() / len(correct_prediction)
    return accuracy

def validation(data_name):
    with torch.no_grad():
        validation_set = dataset[data_name]
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
        print("Evaluating %s: loss %f accuracy %f" % (data_name, average_loss, average_accuracy))
        sys.stdout.flush()

    return average_loss

def train(data_name_list, total_epoch):

    training_set = []
    for data_name in data_name_list:
        training_set = training_set + dataset[data_name]
    print("train %s size %d for %d epoch\n" % (str(data_name_list), len(training_set), total_epoch))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(optimizer)

    for epoch in range(total_epoch):
        training_size = len(training_set)
        batch_count = training_size // BATCH_SIZE
        round_to_batch = batch_count * BATCH_SIZE
        permutation = torch.randperm(training_size)[0:round_to_batch].tolist()

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
            print("%s training epoch %d loss %f accuracy %f\n" % (str(data_name_list), epoch, average_loss, average_accuracy))
            validation("cont_train")
            validation("rand_train")
            validation("cont_valid")
            validation("rand_valid")
            print("saving checkpoint")
            print("")
            torch.save(model, dataset_path + ".pt")
            sys.stdout.flush()
            t_last_print = datetime.now()

t_begin = datetime.now()
t_print = None
validation("cont_train")
validation("rand_train")
validation("cont_valid")
validation("rand_valid")
print("")
train(["cont_train","rand_train"], total_epoch1)
t_end = datetime.now()
tdiff_begin_end = t_end - t_begin
print("time spent total: %s" % str(tdiff_begin_end))
