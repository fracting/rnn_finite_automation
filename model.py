import torch
import torch.nn as nn
import torch.nn.functional as F

class DFA(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, category_size, num_layers, batch_size):
        super(DFA, self).__init__()

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.hidden2category = nn.Linear(hidden_dim, category_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        return (h0, c0)

    def forward(self, sequences):
        embeds = self.embeddings(sequences)
        # TODO - use pack_padded_sequence()
        rnn_out, self.hidden = self.rnn(embeds, self.hidden)
        category_space = self.hidden2category(rnn_out[-1])
        category_scores = F.log_softmax(category_space, dim=1)
        return category_scores
