from torch.utils.data import DataLoader
from model.dataloader import EmbeddingModule
import torch.nn as nn
import numpy as np
import torch

def collate_fn(data):
    seqs, labels = zip(*data)
    return seqs, labels

dataloader = DataLoader(nlp_dataset, batch_size=args.batch_size, \
                        shuffle=False, num_workers=0, collate_fn=collate_fn)

model = EmbeddingModule(len(nlp_dataset.word_to_idx),\
                        args.embedding_dim, args.h_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(args.epoch):
    print('Epoch ' + str(epoch))
    for i, sample in enumerate(dataloader):
        x = torch.LongTensor(sample[0]).to(device)
        y = torch.LongTensor(sample[1]).view(-1).to(device)
        y_pred = model(x)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % 100 == 99:
            print(i, loss.item())
            if i % 100000 == 99:
                with open('./checkpoints/checkpoint_' + str(i) + '.pkl', 'wb') as f:
                    pickle.dump(model, f)
