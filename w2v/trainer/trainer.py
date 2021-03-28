from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle


class Trainer:
    def __init__(self, dataloader, model, criterion, optimizer, config):
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config

    def train():
        for epoch in range(self.config.epoch):
            print('Epoch ' + str(epoch))
            for i, sample in enumerate(self.dataloader):
                x = torch.LongTensor(sample[0]).to(self.config.device)
                y = torch.LongTensor(sample[1]).view(-1).to(self.config.device)
                y_pred = self.model(x)

                loss = self.criterion(y_pred, y)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                if i % 100 == 99:
                    print(i, loss.item())
                    if i % 100000 == 99:
                        with open(
                            '../checkpoints/checkpoint_' + str(i) + '.pkl',
                            'wb',
                        ) as f:
                            pickle.dump(self.model, f)
                        print(
                            '../checkpoints/checkpoint_'
                            + str(i)
                            + '.pkl saved'
                        )


class NegSamplingTrainer:
    def __init__(self, dataloader, config):
        self.dataloader = dataloader
        self.config = config
        self.model = None

        self.target_emb = (
            nn.Embedding(len(dataloader.dataset), config.embed_dim)
            .float()
            .to(config.device)
        )
        self.context_emb = (
            nn.Embedding(len(dataloader.dataset), config.embed_dim)
            .float()
            .to(config.device)
        )
        self.target_emb.weight.data.uniform_(-1, 1)
        self.context_emb.weight.data.uniform_(-1, 1)
        sigmoid = nn.Sigmoid()
        similar = nn.CosineSimilarity()

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(
            list(self.target_emb.parameters())
            + list(self.context_emb.parameters()),
            lr=5e-3,
            momentum=0.9,
        )

    def train():
        for epoch in range(self.config.epoch):
            print('Epoch ' + str(epoch))
            for i, sample in enumerate(self.dataloader):
                pos_idx = sample[0][0].long().to(config.device)
                neg_idx = sample[1][0].long().to(config.device)
                target_idx = sample[0][1].long().to(config.device)
                pos_label = sample[0][2].float().to(config.device)
                neg_label = sample[1][2].float().to(config.device)

                pos = self.context_emb(pos_idx)
                neg = self.context_emb(neg_idx)
                target = self.target_emb(target_idx)

                pred_pos = self.sigmoid(self.similar(pos, target))
                pred_neg = self.sigmoid(
                    self.similar(neg, target.unsqueeze(1).expand_as(neg))
                )

                pos_loss = criterion(pred_pos, pos_label)
                neg_loss = torch.sum(
                    criterion(
                        pred_neg, neg_label.unsqueeze(1).expand_as(pred_neg)
                    )
                )

                pos_loss.retain_grad()
                neg_loss.retain_grad()

                self.loss = pos_loss + neg_loss
                self.loss.retain_grad()

                self.optimizer.zero_grad()
                self.loss.backward(retain_graph=True)
                self.optimizer.step()

                if i % 100 == 99:
                    print(i, loss.item())
                    if i % 100000 == 99:
                        with open(
                            './checkpoints/neg_sample_checkpoint_epoch'
                            + str(epoch)
                            + '_'
                            + str(i)
                            + '.pkl',
                            'wb',
                        ) as f:
                            pickle.dump(self.target_emb, f)
                        print(
                            './checkpoints/neg_sample_checkpoint_epoch'
                            + str(epoch)
                            + '_'
                            + str(i)
                            + '.pkl saved'
                        )
                self.model = self.target_emb
