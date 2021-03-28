import torch.nn as nn


class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, config):
        super(self).__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim).float()
        self.embedding.weight.data.uniform_(-1, 1)
        self.linear1 = nn.Linear(
            config.embed_dim * config.window_size, config.h_dim
        )
        self.linear1.weight.data.uniform_(-1, 1)
        self.linear2 = nn.Linear(config.h_dim, vocab_size)
        self.linear2.weight.data.uniform_(-1, 1)
        self.motorway = nn.Linear(
            config.embed_dim * config.window_size, vocab_size
        )
        self.motorway.weight.data.uniform_(-1, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        embedded = self.embedding(x).view(x.shape[0], -1)
        embedded.retain_grad()
        net = self.linear1(embedded)
        net = self.tanh(net)
        net = self.linear2(net)
        net = net + self.motorway(embedded)
        return net
