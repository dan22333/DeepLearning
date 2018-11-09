import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import csv
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
import matplotlib.pyplot as plt


class ReQU(torch.nn.Module):

    def __init__(self):
        super(ReQU, self).__init__()

    def forward(self, input):
        self.save_for_backward = input
        temp = input.clamp(min=0)
        result = torch.pow(temp,2)
        return result


class FlowerDataset(Dataset):
    """Flower dataset."""
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.flowers = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.flowers)

    def __getitem__(self, idx):
        sample_x = self.flowers.ix[idx, :3].as_matrix().astype('float')
        types = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
        sample_y = types[self.flowers.ix[idx, 4]]
        if self.transform:
            sample = self.transform(sample)

        return sample_x,sample_y

# batch size
batch_s= 25# input dimension
D_in = 4
# hidden layer dimension
H = 4
# output dimension
D_out = 3

Flower_dataset = FlowerDataset(csv_file='iris.data.csv')
dataloader = DataLoader(Flower_dataset, batch_size=batch_s,
                        shuffle=True)


# define model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    ReQU(),
    torch.nn.Linear(H, D_out),
    torch.nn.LogSoftmax()
)

# define loss function
loss_fn = torch.nn.NLLLoss(size_average=False)

# define optimizer
test_threshold = 1e-3
learning_rate = 1e-4
epochs = 2000
train_prints_count = 10
train_prints_interval = epochs / train_prints_count
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.double()
print('############ Train #############')
train_losses = []
for t in range(epochs):
    batch_loss_accumulator = 0
    last_batch = 0
    if t % train_prints_interval == 0:
        print('Epoch =' + str(t))
    for i_batch, sample_batched in enumerate(dataloader):
        last_batch = i_batch
        x = Variable(sample_batched[0])
        y = Variable(sample_batched[1], requires_grad=False)
        # forward pass
        y_pred = model(x)
        # calculate and print loss
        loss = loss_fn(y_pred, y)
        if t % train_prints_interval == 0:
            print('Batch = ' + str(i_batch) + ' LOSS=' + str(loss.data[0]))
        batch_loss_accumulator += loss.data[0]

        # zero gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # optimizer step
        optimizer.step()
    train_loss = batch_loss_accumulator/(last_batch+1)
    train_losses.append(train_loss)     # train loss is the average of the batch losses in the epoch
print('Done.')
plt.plot(np.arange(len(train_losses)), np.array(train_losses), label='Train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print('\n############ Test #############')
correct_examples=0
y_test = y.data.numpy()
for i in range(len(y_test)):
    prediction = model(x[i])
    _, inx = torch.topk(prediction,1,-1,True)
    if y_test[i] == inx.data.numpy()[0]:
        correct_examples+=1
    


print('\nAccuracy on traning set = ' + str(100.0*correct_examples/len(y)) + '%')
