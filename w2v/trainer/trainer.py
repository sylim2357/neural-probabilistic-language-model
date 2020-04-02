from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle


class Trainer():
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
                        with open('../checkpoints/checkpoint_' + str(i) + '.pkl', 'wb') as f:
                            pickle.dump(self.model, f)
                        print('../checkpoints/checkpoint_' + str(i) + '.pkl saved')
