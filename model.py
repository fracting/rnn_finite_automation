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

        self.hidden = self.init_hidden()

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
        embeds = self.embeddings(sequences)
        # TODO - use pack_padded_sequence()
        seq_len = len(embeds)
        hiddens = []
        for i in range(seq_len):
            elem = embeds[i]
            self.hidden = self.rnnCell(elem, self.hidden)
            hiddens.append(self.hidden)

        if self.rnn_type == "RNN":
            hn = self.hidden
        elif self.rnn_type == "GRU":
            hn = self.hidden
        elif self.rnn_type == "LSTM":
            hn, _ = self.hidden
        else:
            raise NotImplementedError("rnn_type not recognized")

        category_space = self.hidden2category(hn)
        category_scores = F.log_softmax(category_space, dim=1)
        return category_scores, hiddens
