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
            rnn = nn.RNN
        elif rnn_type == "GRU":
            rnn = nn.GRU
        elif rnn_type == "LSTM":
            rnn = nn.LSTM
        else:
            raise NotImplementedError("rnn_type not recognized")
        self.rnn = rnn(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.hidden2category = nn.Linear(hidden_dim, category_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
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
        rnn_out, self.hidden = self.rnn(embeds, self.hidden)
        category_space = self.hidden2category(rnn_out[-1])
        category_scores = F.log_softmax(category_space, dim=1)
        return category_scores
