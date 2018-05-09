import torch
import torch.nn as nn
import torch.nn.functional as F

class DFA(nn.Module):

    def __init__(self, rnn_type, embedding_dim, hidden_dim, vocab_size, category_size, num_layers, batch_size, dropout):

        print("rnn_type: %s" % rnn_type)
        print("embedding dim: %d" % embedding_dim)
        print("hidden dim: %d" % hidden_dim)
        print("vocab size: %d" % vocab_size)
        print("category size: %d" % category_size)
        print("num layers: %d" % num_layers)
        print("batch size: %d" % batch_size)
        print("dropout: %f" % dropout)
        print("")

        super(DFA, self).__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if rnn_type == "RNN":
            self.rnnCell = nn.RNNCell(embedding_dim, hidden_dim, nonlinearity='relu')
        elif rnn_type == "GRU":
            self.rnnCell = nn.GRUCell(embedding_dim, hidden_dim)
        elif rnn_type == "LSTM":
            self.rnnCell = nn.LSTMCell(embedding_dim, hidden_dim)
        else:
            raise NotImplementedError("rnn_type not recognized")
        self.hidden2category = nn.Linear(hidden_dim, category_size)

    def init_hidden(self):
        h0 = torch.zeros(self.batch_size, self.hidden_dim)
        c0 = torch.zeros(self.batch_size, self.hidden_dim)
        if self.rnn_type == "RNN":
            hidden = h0
        elif self.rnn_type == "GRU":
            hidden = h0
        elif self.rnn_type == "LSTM":
            hidden = (h0, c0)
        else:
            raise NotImplementedError("rnn_type not recognized")

        return hidden

    def forward(self, sequences):
        # TODO - use pack_padded_sequence()
        hiddens = []
        hidden = self.init_hidden()
        for batch_elems_onehot in sequences:
            hidden = self.rnnCell(batch_elems_onehot, hidden)
            hiddens.append(hidden)

        if self.rnn_type == "RNN":
            hn = hidden
        elif self.rnn_type == "GRU":
            hn = hidden
        elif self.rnn_type == "LSTM":
            hn, _ = hidden
        else:
            raise NotImplementedError("rnn_type not recognized")

        category_space = self.hidden2category(hn)
        category_scores = F.log_softmax(category_space, dim=1)
        return category_scores, hiddens
