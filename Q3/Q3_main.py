import os
import csv
import torch
from torch.utils.data.dataloader import DataLoader
from face_net_part3 import FaceNetP3
from face_net_part4 import FaceNetP4
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from preprocess import load_train_data, load_test_data

ConfP3 = {'model_class': FaceNetP3, 'epochs': 600, 'learning_rate': 1e-4, 'suffix': '600_epochs_P3'}
ConfP4 = {'model_class': FaceNetP4, 'epochs': 150, 'learning_rate': 1e-4, 'suffix': '150_epochs_P4'}

CONFIGURATION = ConfP4

TRAIN = True
TEST_ON_REAL_IMAGES = False

TRAIN_FILENAME = os.path.join('data', 'training.csv')
TEST_FILENAME = os.path.join('data', 'test.csv')
MODEL_FILENAME = 'trained_model_' + CONFIGURATION['suffix'] + '.pt'
PICKLE_FILENAME = 'trained_model_' + CONFIGURATION['suffix'] + '.pkl'
LOG_CSV_FILENAME = 'loss_curves_' + CONFIGURATION['suffix'] + '.csv'

BATCH_SIZE = 128
TEST_SET_PROPORTION = 0.20
LEARNING_RATE = CONFIGURATION['learning_rate']
MAX_EPOCHS = CONFIGURATION['epochs']

PIXEL_MAX_VAL = 255.0
COORDINATE_MAX_VAL = 96.0
NUM_EXAMPLES_TO_SHOW = 20

TEST_IMAGES = ['test2.jpg']

model = CONFIGURATION['model_class']()

loss_fn = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def plot_image_and_predictions(img, predictions):
    xs = []
    ys = []
    plt.figure()
    plt.imshow(img, cmap='gray')
    for coord in range(0,30,2):
        xs.append(predictions[coord])
        ys.append(predictions[coord+1])
    plt.scatter(xs,ys)
    plt.show()


if TRAIN:
    # load data
    X, y = load_train_data(TRAIN_FILENAME)

    # split to test/set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_PROPORTION, shuffle=True, random_state=42)

    # wrap in tensors & variables
    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_train_var, y_train_var = Variable(X_train), Variable(y_train)
    X_test_var, y_test_var = Variable(torch.FloatTensor(X_test)), Variable(torch.FloatTensor(y_test))
    TEST_SET_SIZE = len(X_test)

    # init dataloader
    train_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)

    # main loop
    start_time = time.time()
    train_losses = []
    test_losses = []
    with open(LOG_CSV_FILENAME, 'w',newline='') as log_csv_file:
        field_names = ['EPOCH', 'TRAIN_LOSS', 'TEST_LOSS']
        csv_writer = csv.writer(log_csv_file, delimiter=',')
        csv_writer.writerow(field_names)

    for epoch in range(MAX_EPOCHS):
        batch_loss_accumulator = 0
        last_batch = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_dataset):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            last_batch = batch_idx
            # predict
            output = model(batch_x)

            # calc loss
            loss = loss_fn(output, batch_y)
            batch_loss_accumulator += loss.data[0]

            # step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('EPOCH=' + str(epoch) + '   BATCH=' + str(batch_idx) + '   BATCH LOSS=' + str(loss.data[0]))

        # ---------------------------- #
        #           evaluate           #
        # ---------------------------- #
        train_loss = batch_loss_accumulator/(last_batch+1)
        train_losses.append(train_loss)     # train loss is the average of the batch losses in the epoch
        test_pred = model(X_test_var)
        test_loss = loss_fn(test_pred, y_test_var)
        test_loss = test_loss.data[0]
        test_losses.append(test_loss)

        print('@@@ train loss=' + str(float(train_loss)) + '  @@@ test loss=' + str(float(test_loss)))

        with open(LOG_CSV_FILENAME, 'a',newline='') as log_csv_file:
            csv_writer = csv.writer(log_csv_file, delimiter=',')
            csv_writer.writerow([str(epoch), str(float(train_loss)), str(float(test_loss))])

    end_time = time.time()
    print('---------------------------------------------------------')
    print('Time elapsed=' + str(end_time - start_time) + ' seconds.')
    log_csv_file.close()
    torch.save(model.state_dict(), MODEL_FILENAME)
    with open(PICKLE_FILENAME, 'wb') as f:
        pickle.dump(model, f)
    print('Done.')
    training_data = np.genfromtxt(LOG_CSV_FILENAME, delimiter=',')
    plt.plot(training_data[1:, 0], training_data[1:, 1], label='Train loss')
    plt.plot(training_data[1:, 0], training_data[1:, 2], label='Test loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

# TEST MODE
else:
    with open(PICKLE_FILENAME, 'rb') as f:
        model = pickle.load(f)
    if not TEST_ON_REAL_IMAGES:
        test_images = load_test_data(TEST_FILENAME)
        for i in range(NUM_EXAMPLES_TO_SHOW):
            image = test_images[i][0]
            img_tensor = torch.FloatTensor(image)
            img_tensor = img_tensor.view(1, 1, 96, 96)
            output = model(img_tensor)
            output = output.data[0].numpy()
            output = (output * 48.0) + 48.0
            plot_image_and_predictions(image, output)
    else:
        img = Image.open(TEST_IMAGES[0]).convert('L')
        img = img.resize([96, 96], Image.ANTIALIAS)
        img_data = np.asarray(img.getdata()).reshape(img.size)
        img_data = img_data / float(PIXEL_MAX_VAL)  # normalize to 0..1
        img_tensor = torch.FloatTensor(img_data)
        img_tensor = img_tensor.view(1, 1, 96, 96)

        output = model(img_tensor)
        output = output.data[0].numpy()
        output = output * 48.0 + 48.0

        plot_image_and_predictions(img_data, output)

print('Done.')