import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense , Input , LSTM, Bidirectional, Dropout
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras import regularizers


def wavelet_transform(df):
    ca , cb, cc , cd = pywt.wavedec(df['price'].values, 'haar', level = 3)
    cat = pywt.threshold(ca, np.std(ca), mode = 'soft')
    cbt = pywt.threshold(cb, np.std(cb), mode = 'soft')
    cct = pywt.threshold(cc, np.std(cc), mode = 'soft')
    cdt = pywt.threshold(cd, np.std(cd), mode = 'soft')
    coeff = [cat , cbt, cct , cdt]
    return pywt.waverec(coeff, 'haar')

def get_sample(df, length, temporal_horizon):

    temporal_horizon = temporal_horizon - 1
    last_possible = df.shape[0] - temporal_horizon - length
    random_start = np.random.randint(1, last_possible)
    X_sample = df.drop(columns = 'price').iloc[random_start: random_start+length].values
    y_sample = df['price'].iloc[random_start+length: random_start+length+temporal_horizon+1].values

   # if y_sample != y_sample:
        #X_sample, y_sample = get_sample(df, length, temporal_horizon)

    return X_sample, y_sample

def get_X_y(df, temporal_horizon, length_of_sequences):
    X, y = [], []

    for len_ in length_of_sequences:
        xi, yi = get_sample(df, len_, temporal_horizon)
        X.append(xi)
        y.append(yi)

    return X, y


def autoencoder(features):
    input_data = Input(shape=(1, features))
    encoded1 = Dense(features, activation="relu", activity_regularizer=regularizers.l2(0))(input_data)
    one_l = Dense(1, activation="relu", activity_regularizer=regularizers.l2(0))(encoded1)
    decoded = Dense(features, activation="linear", activity_regularizer=regularizers.l2(0))(one_l)
    autoencoder = Model(inputs=input_data, outputs=decoded)
    encoder = Model(input_data, one_l)
    autoencoder.compile(loss = 'mse', optimizer = 'rmsprop',metrics = ['mae'])
    return autoencoder , encoder

def init_model(length, n_days) :
    model = Sequential()
    model.add(LSTM(150,activation = 'tanh',input_shape=(length, 14),return_sequences = True))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(120, activation = 'tanh',return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(100,activation = 'tanh',return_sequences = True)))
    model.add(Dropout(0.5))
    model.add(LSTM(80,activation = 'tanh'))
    model.add(Dense(60,activation = 'relu'))
    model.add(Dense(n_days,activation = 'linear'))

    model.compile(loss = 'mse', optimizer = 'rmsprop',metrics = ['mae'])

    return model

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Mean Square Error - Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()

    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model loss')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()
