import torch
from torch.autograd import Variable

# training data
training_x = torch.FloatTensor([[1,1], [1,0], [0,1], [0,0]])
training_y = torch.FloatTensor([0, 1, 1, 0])

# batch size
N = 1
# input dimension
D_in = 2
# hidden layer dimension
H = 3
# output dimension
D_out = 1


# init input/output tensors
x = Variable(training_x)
y = Variable(training_y, requires_grad=False)


# define model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)

# define loss function
loss_fn = torch.nn.MSELoss(size_average=True)

# define optimizer
test_threshold = 1e-3
learning_rate = 1e-4
epochs = 30000
train_prints_count = 30
train_prints_interval = epochs / train_prints_count
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print('############ Train #############')

for t in range(epochs):
    # forward pass
    y_pred = model(x)

    # calculate and print loss
    loss = loss_fn(y_pred, y)
    if t % train_prints_interval == 0:
        print('EPOCH = ' + str(t) + ' LOSS=' + str(loss.data[0]))

    # zero gradients
    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # optimizer step
    optimizer.step()


print('\n############ Test  #############')
num_of_train_examples = 4
correct_examples = 0

for i in range(num_of_train_examples):
    prediction = model(x[i])
    print('Prediction on ' + str(round(x[i].data[0])) + ',' + str(round(x[i].data[1])) + ' = ' + str(round(prediction.data[0])))
    if abs(training_y[i] - prediction.data[0]) < test_threshold:
        correct_examples += 1
        print('\tCorrect!')
    else:
        print('\tIncorrect!')

print('\nAccuracy = ' + str(100.0*correct_examples/num_of_train_examples) + '%')
