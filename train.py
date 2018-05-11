import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from datetime import datetime
import sys
import random
import copy

from model import DFA
from data import char_to_ix, category_to_ix, seqs_to_tensor, categories_to_tensor, load_dataset, train_embedding
from util import semantics_loss_fn
from create_dataset import classify

RNN_TYPE = "RNN"
HIDDEN_DIM = 80
NUM_LAYERS = 1
BATCH_SIZE = 128
EMBEDDING_DIM = 20
DROPOUT = 0.0 # dropout does not apply on output layer, so no effect to single layer network

print_per_epoch = 10
update_per_counter = 10
total_epoch1 = 3000
print("total_epoch1 %d" % total_epoch1)

torch.manual_seed(4) # TODO - disable manual seed in production version

cont_train_size = 0
#rand_train_size = BATCH_SIZE
rand_train_size = 20000 # super large
cont_valid_size = 4096
rand_valid_size = 4096
generate_input_iters = BATCH_SIZE
class_type = "imbalance"
divider = 7
dataset_name = "10div" + str(divider) + "." + class_type
dataset_path = "dataset/" + dataset_name + ".txt"
dataset, vocab_size, category_size = load_dataset(dataset_path, cont_train_size, rand_train_size, cont_valid_size, rand_valid_size)

print("prepare embedding")
embedding_model = train_embedding(vocab_size, EMBEDDING_DIM, BATCH_SIZE)

dataset["dyna_train"] = []

load_model = False
model_name = dataset_name
write_model_path = "checkpoint/" + model_name + ".pt"
read_model_path = write_model_path
hidden_csv_path = model_name + ".csv"
if load_model:
    print("read_model_path: " + read_model_path)
    model = torch.load(read_model_path)
else:
    model = DFA(RNN_TYPE, EMBEDDING_DIM, HIDDEN_DIM, len(char_to_ix), category_size, NUM_LAYERS, BATCH_SIZE, DROPOUT)
learning_rate = 0.001

loss_function = nn.NLLLoss(reduce = False)

def calc_accuracy(score_tensors, target):
    _, index_tensors = score_tensors.max(dim=1)
    correct_prediction = (index_tensors == target)
    accuracy = correct_prediction.sum().item() / len(correct_prediction)
    return accuracy

def generate_new_input(old_input, targets):
    onehot_seqs = copy.deepcopy(old_input)
    onehot_seqs.requires_grad_()

    all_seq_batches = []
    all_seq_batches_int = []
    for i in range(generate_input_iters):
        input_optimizer = optim.Adam([onehot_seqs], lr=learning_rate * 100)
        input_optimizer.zero_grad()

        embedding_seqs = []
        for onehot_elems in onehot_seqs:
            out, embedding_elems = embedding_model(onehot_elems)
            embedding_seqs.append(embedding_elems)
        embedding_seqs = torch.stack(embedding_seqs)
        category_scores, _ = model.forward(embedding_seqs)
        category = torch.exp(category_scores)

        batch_loss = loss_function(category_scores, targets)
        reduced_batch_loss = batch_loss.sum() / BATCH_SIZE
        negative_reduced_batch_loss = - reduced_batch_loss

        in_semantics_loss = torch.sum(semantics_loss_fn(onehot_seqs, dim=2))
        combined_loss = negative_reduced_batch_loss #+ in_semantics_loss / 1000
        combined_loss.backward()
        input_optimizer.step()
        # Hack: using .data to  workaround ValueError("can't optimize a non-leaf Tensor")
        onehot_seqs = torch.clamp(onehot_seqs, 1e-7, 1 - 1e-7).data.requires_grad_()
        onehot_seqs_argmax = torch.argmax(onehot_seqs, dim=2)
        seq_batches = onehot_seqs_argmax.transpose(0,1).tolist()
        seq_batches_tuple = [tuple(str(x) for x in seq) for seq in seq_batches]
        print("seq_batches_tuple[:5]", *seq_batches_tuple[:5], sep="\n")
        all_seq_batches = list(set(all_seq_batches + seq_batches_tuple))

    print("combined_loss:\n", combined_loss)
    all_seq_batches_int = [int("".join(seq)) for seq in all_seq_batches]
    new_targets = [str(classify(input, divider, class_type)) for input in all_seq_batches_int]
    new_dataset = list(zip(all_seq_batches, new_targets))
    # filter out "Other" type
    new_dataset = [line for line in new_dataset if line[1] != "Other"]
    return new_dataset

def validation(data_name, dump_hidden, update_dataset):
    with torch.no_grad():
        validation_set = dataset[data_name]
        validation_size = len(validation_set)
        batch_count = validation_size // BATCH_SIZE
        round_to_batch = batch_count * BATCH_SIZE

        validation_loss = 0
        validation_accuracy = 0
        hidden_dump = ""
        validation_set.sort(key = lambda x: len(x[0]))
        validation_set = validation_set[:round_to_batch]
        skipped_batch = 0
        training_cache = []
        for i in range(0, round_to_batch, BATCH_SIZE):
            model.zero_grad()

            batch_data = validation_set[i:i+BATCH_SIZE]
            if len(batch_data[0][0]) != len(batch_data[-1][0]):
                # give up batch with inconsistent seq len
                skipped_batch = skipped_batch + 1
                continue

            seqs, categories = list(zip(*batch_data))
            seqs = list(seqs)
            categories = list(categories)
            onehot_seqs, embedding_seqs = seqs_to_tensor(seqs, char_to_ix, vocab_size, EMBEDDING_DIM, embedding_model)
            targets = categories_to_tensor(categories, category_to_ix)

            category_scores, hiddens = model(embedding_seqs)
            batch_loss = loss_function(category_scores, targets)
            category = torch.exp(category_scores)
            perplexity = torch.exp(-torch.sum(category * torch.log(category), dim=1)).tolist()
            semantics_loss = semantics_loss_fn(category, dim=1)

            reduced_batch_loss = batch_loss.sum() / BATCH_SIZE
            validation_loss = validation_loss + float(reduced_batch_loss)

            batch_accuracy = calc_accuracy(category_scores, targets)
            validation_accuracy = validation_accuracy + batch_accuracy

            if dump_hidden:
                if model.rnn_type == "LSTM":
                    cols = len(hiddens[0][0][0]) * 2
                    for (hs, cs) in hiddens:
                        hs = hs.tolist()
                        cs = cs.tolist()
                        for (h, c) in list(zip(hs, cs)):
                            line = "id" + ", " + str(h)[1:-1] + ", " + str(c)[1: -1] + "\n"
                            hidden_dump = hidden_dump + line
                else:
                    cols = len(hiddens[0][0])
                    for hs in hiddens:
                        hs = hs.tolist()
                        for h in hs:
                            line = "id" + ", " + str(h)[1:-1] + "\n"
                            hidden_dump = hidden_dump + line

        if dump_hidden:
            print("write hidden values to csv %s" % hidden_csv_path)
            hidden_csv = open(hidden_csv_path, "w+")
            hidden_csv.write(" " + ", v" * cols + "\n")
            hidden_csv.write(hidden_dump)
            hidden_csv.close()

        average_loss = validation_loss / (batch_count - skipped_batch)
        average_accuracy = validation_accuracy / (batch_count - skipped_batch)
        print("Evaluating %s: loss %f accuracy %f" % (data_name, average_loss, average_accuracy))
        training_cache.sort(reverse = False, key = lambda x: x[3])
        print(*training_cache[-3:], sep="\n")
        #if counter % update_per_counter == 0:
        if update_dataset:
            None
            #print("update training set")
            #dataset["dyna_train"] = dataset["dyna_train"] + list(list(zip(*training_cache[-512-64:]))[0])
        sys.stdout.flush()

    return average_loss

def train(data_name_list, total_epoch):

#    training_set = []
#    for data_name in data_name_list:
#        training_set = training_set + dataset[data_name]
#    print("train %s size %d for %d epoch\n" % (str(data_name_list), len(training_set), total_epoch))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(optimizer)

    last_average_accuracy = 0
    for epoch in range(total_epoch):

        #####
        training_set = []
        for data_name in data_name_list:
            training_set = training_set + dataset[data_name]
        #####

        training_size = len(training_set)
        batch_count = training_size // BATCH_SIZE
        round_to_batch = batch_count * BATCH_SIZE
        permutation = torch.randperm(training_size)[0:round_to_batch].tolist()
        training_set = [training_set[index] for index in permutation]
        training_set.sort(key = lambda x: len(x[0]))

        epoch_loss = 0
        epoch_accuracy = 0
        skipped_batch = 0
        dataset["tmp"] = []
        for i in range(0, round_to_batch, BATCH_SIZE):
            model.zero_grad()

            batch_data = training_set[i:i+BATCH_SIZE]
            if len(batch_data[0][0]) != len(batch_data[-1][0]):
                # give up batch with inconsistent seq len
                skipped_batch = skipped_batch + 1
                continue

            seqs, categories = list(zip(*batch_data))
            seqs = list(seqs)
            categories = list(categories)
            onehot_seqs, embedding_seqs = seqs_to_tensor(seqs, char_to_ix, vocab_size, EMBEDDING_DIM, embedding_model)
            targets = categories_to_tensor(categories, category_to_ix)

            category_scores, hiddens = model(embedding_seqs)

            batch_loss = loss_function(category_scores, targets)
            reduced_batch_loss = batch_loss.sum() / BATCH_SIZE
            epoch_loss = epoch_loss + float(reduced_batch_loss)
            reduced_batch_loss.backward()
            optimizer.step()

            batch_accuracy = calc_accuracy(category_scores, targets)
            epoch_accuracy = epoch_accuracy + batch_accuracy
            size = len(training_set)
            if last_average_accuracy > 0.95 and size < 16382:
                print("batch_accuracy", batch_accuracy, "i", i)
                dataset["tmp"] = dataset["tmp"] + generate_new_input(onehot_seqs, targets)

        average_loss = epoch_loss / (batch_count - skipped_batch)
        average_accuracy = epoch_accuracy / (batch_count - skipped_batch)
        last_average_accuracy = average_accuracy

        dataset_uniq = set(dataset["dyna_train"] + dataset["tmp"])
        dataset_uniq_list = list(dataset_uniq)
        dataset["dyna_train"] = dataset_uniq_list


        if epoch % print_per_epoch == 0:
            t_print = datetime.now()
            if epoch > 1:
                t_diff_per_print = t_print - t_last_print
                print("time spent in %d epoch %s" % (print_per_epoch, str(t_diff_per_print)))

            print("training %s, size %d, epoch %d, total %d, loss %f accuracy %f\n" % (str(data_name_list), len(training_set), epoch, total_epoch, average_loss, average_accuracy))
            #validation("rand_train", True)
            #validation("cont_train", False)
            #validation("rand_valid", False)
            validation("cont_valid", False, average_accuracy > 0.95)
            validation("rand_valid", False, average_accuracy > 0.95) # haha todo change back to 0.95 TODO FIXME HACK WRONG
            print("saving checkpoint")
            print("")
            torch.save(model, write_model_path)
            sys.stdout.flush()
            t_last_print = datetime.now()

t_begin = datetime.now()
t_print = None
validation("cont_valid", False, False)
print("")
#train(["cont_train","dyna_train"], total_epoch1)
dataset["rand_train128"] = dataset["rand_train"] * 128
#train(["rand_train128", "dyna_train"], total_epoch1)
train(["rand_train"], total_epoch1)
#train(["cont_train", "rand_train"], total_epoch1)
#train(["cont_train"], total_epoch1)
t_end = datetime.now()
tdiff_begin_end = t_end - t_begin
print("time spent total: %s" % str(tdiff_begin_end))
