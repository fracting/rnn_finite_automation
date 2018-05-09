import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from data import onehot

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
        embedding = F.relu(embedding)
        output = self.embedding2output(embedding)
        output = F.log_softmax(output, dim=1)
    
        return output, embedding

learning_rate = 0.1
loss_function = nn.NLLLoss()
vocab_size = 10
embedding_dim = 80
batch_size = 128

def train_embedding(vocab_size, embedding_dim, batch_size):
    input = torch.randint(0, batch_size, (batch_size,)).long()
    input = input % vocab_size
    input_onehot = onehot(input, vocab_size)

    for epoch in range(0, 5):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.zero_grad()
        output, embedding = model(input_onehot)
        loss = loss_function(output, input)
        loss.backward()
        optimizer.step()

model = embed(vocab_size, embedding_dim, batch_size)
train_embedding(vocab_size, embedding_dim, batch_size)
