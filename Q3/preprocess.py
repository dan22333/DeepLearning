# Preprocess and load data

import csv
import numpy as np


def load_train_data(filename, load_only_full_samples=True):
    PIXEL_MAX_VAL = 255.0
    COORDINATE_MAX_VAL = 96.0
    data_x = []
    data_y = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)   # skip headers
        for line in reader:
            labels_data = []
            curr_sample_is_full = True

            # process coordinates
            for coordinate in line[:30]:
                if coordinate == '':
                    curr_sample_is_full = False
                    labels_data.append(None)
                else:
                    normalized_coordinate = 2.0*(float(coordinate)/COORDINATE_MAX_VAL) - 1
                    labels_data.append(normalized_coordinate)

            # process image and append if applicable
            if (load_only_full_samples and curr_sample_is_full) or (not load_only_full_samples):
                image_data = np.asarray([float(pixel_value)/PIXEL_MAX_VAL for pixel_value in line[30].split(sep=' ')])
                image_data = np.reshape(image_data, [1, 96, 96])
                data_x.append(image_data)
                data_y.append(labels_data)

    return data_x, data_y



def load_test_data(filename, limit=None):
    PIXEL_MAX_VAL = 255.0
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip headers
        for line in reader:
            image = line[1].split(sep=' ')
            image = [float(pixel)/PIXEL_MAX_VAL for pixel in image]
            image = np.asarray(image)
            image = np.reshape(image, [1, 96, 96])
            data.append(image)
    return data

