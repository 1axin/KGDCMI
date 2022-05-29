os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # cpu
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math
import random
def ReadCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return
def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    retur
SampleFeature = []
ReadMyCsv(SampleFeature, "circSample.csv")
print(len(SampleFeature))
SampleFeature = np.array(SampleFeature)
print('SampleFeature',len(SampleFeature))
print('SampleFeature[0]',len(SampleFeature[0]))
x = SampleFeature #
x_train = SampleFeature
x_test = SampleFeature
print(x_train.shape)
print(x_test.shape)
print(type(x_train[0][0]))
encoding_dim = 128
input_img = Input(shape=(len(SampleFeature[0]),))
from keras import regularizers
encoded_input = Input(shape=(encoding_dim,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-7))(input_img)    # 与单层的唯一区别 (from keras import regularizers)!!!注意调节参数10e-7
decoded = Dense(512, activation='sigmoid')(encoded)
autoencoder = Model(inputs=input_img, outputs=decoded)
decoder_layer = autoencoder.layers[-1]
encoder = Model(inputs=input_img, outputs=encoded)
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=50, shuffle=True, validation_data=(x_test, x_test))
encoded_imgs = encoder.predict(x)
decoded_imgs = decoder.predict(encoded_imgs)
print(len(encoded_imgs))
print(len(encoded_imgs[1]))
RNAname = []
read_csv1(RNAname,'circrna_1024_Kmer.csv')
RNAname = np.array(RNAname)
RNAname = RNAname[:,np.newaxis]
matrix = []
matrix = np.hstack((RNAname,encoded_imgs))
storFile(matrix, 'circSampleFeature.csv')