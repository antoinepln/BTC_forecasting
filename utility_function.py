import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense , Input , LSTM, Bidirectional, Dropout
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, confusion_matrix
import ipdb



def get_sample(df, length, temporal_horizon, style, start):

    temporal_horizon = temporal_horizon - 1
    last_possible = df.shape[0] - temporal_horizon - length

    if style == "linear" :
        X_sample = df.drop(columns = 'price').iloc[start: start+length].values
        y_sample = df['price'].iloc[start+length: start+length+temporal_horizon+1].values
        return X_sample, y_sample
    if style == "clf" :
        X_sample = df.drop(columns = 'up').iloc[start: start+length].values
        y_sample = df['up'].iloc[start+length+temporal_horizon-1: start+length+temporal_horizon].values
        return X_sample, y_sample


def get_X_y(df, temporal_horizon, length, style):
    X, y = [], []

    for len_ in range(len(df)-temporal_horizon -length):
        xi, yi = get_sample(df, length, temporal_horizon, style , len_)
        X.append(xi)
        y.append(yi)

    return X, y



def get_X_y_DNN(df, temporal_horizon, length, style):
    X, y = [], []

    for len_ in range(len(df)-temporal_horizon -length):
        xi, yi = get_sample(df, length, temporal_horizon, style , len_)
        X.append(xi.ravel())
        y.append(yi.ravel())

    return X, y

def wavelet_transform(df):
    ca , cb = pywt.wavedec(df.values, 'haar', level = 1)
    cat = pywt.threshold(ca, np.std(ca), mode = 'soft')
    cbt = pywt.threshold(cb, np.std(cb), mode = 'soft')
    coeff = [cat , cbt ]
    return pywt.waverec(coeff, 'haar')

def autoencoder(features):
    input_data = Input(shape=(1, features))
    encoded1 = Dense(features, activation="relu", activity_regularizer=regularizers.l2(0))(input_data)
    one_l = Dense(1, activation="relu", activity_regularizer=regularizers.l2(0))(encoded1)
    decoded = Dense(features, activation="linear", activity_regularizer=regularizers.l2(0))(one_l)
    autoencoder = Model(inputs=input_data, outputs=decoded)
    encoder = Model(input_data, one_l)
    autoencoder.compile(loss = 'mse', optimizer = 'rmsprop',metrics = ['mae'])
    return autoencoder


def lstm_model(length, n_days, features, style) :
    model = Sequential()
    model.add(LSTM(features,activation = 'tanh',input_shape=(length, features),return_sequences = True))
    model.add(Bidirectional(LSTM(features/2,activation = 'tanh')))
    model.add(Dropout(0.2))

    if style == 'linear' :
        model.add(Dense(n_days,activation = 'linear'))

        model.compile(loss = 'mse', optimizer = 'rmsprop',metrics = ['mae'])
    if style == 'clf':
        model.add(Dense(1,activation = 'sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def DNN_model(length, n_days, features) :
    nb = length * features
    model = Sequential()
    model.add(Dense(nb,activation = 'relu',input_dim=(nb)))
    model.add(Dense(nb/2,activation = 'relu'))
    model.add(Dense(nb/4,activation = 'relu'))
    model.add(Dense(1,activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Mean Square Error - Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model loss')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()



def get_accuracy_LSTM(df_test, df_true, model, length) :
    pred = []
    profit = 1
    mini = 1
    maxi = 1
    df_true['close'] = df_true['close'].apply(lambda x : np.exp(x))
    for x in range(len(df_test)-length):
        y = df_test.drop(columns = 'up')[x:x+length].values
        y.shape = (1, y.shape[0],y.shape[1])
        predict = model.predict(y)
        if predict > 0.5 :
            pred.append(1)
        else :
            pred.append(0)

        if x != 0 :
            if pred[x-1] == 1 :

                result = ((df_true['close'].iloc[x] - df_true['close'].iloc[x-1]) / df_true['close'].iloc[x-1]) + 1
                profit = profit * result
                if result < mini :
                    mini = result
                if maxi < result :
                    maxi = result

    accuracy = accuracy_score(df_test['up'][length:].values, np.array(pred).ravel())

    tn, fp, fn, tp = confusion_matrix(df_test['up'][length:].values, np.array(pred).ravel()).ravel()

    recall = tp/(tp+fn)
    speci = tn/(tn+fp)

    return accuracy, recall, speci, profit, mini, maxi

def get_accuracy_DNN(df_test,df_true, model, length) :
    pred = []
    profit = 1
    mini = 1
    maxi = 1
    df_true['close'] = df_true['close'].apply(lambda x : np.exp(x))
    for x in range(len(df_test)-length):
        y = df_test.drop(columns = 'up')[x:x+length].values.ravel()
        y.shape = (1, y.shape[0])
        predict = model.predict(y)
        if predict > 0.3 :
            pred.append(1)
        else :
            pred.append(0)

        if x != 0 :
            if pred[x-1] == 1 :

                result = ((df_true['close'].iloc[x] - df_true['close'].iloc[x-1]) / df_true['close'].iloc[x-1]) + 1
                profit = profit * result
                if result < mini :
                    mini = result
                if maxi < result :
                    maxi = result


    accuracy = accuracy_score(df_test['up'][length:].values, np.array(pred).ravel())

    tn, fp, fn, tp = confusion_matrix(df_test['up'][length:].values, np.array(pred).ravel()).ravel()

    recall = tp/(tp+fn)
    speci = tn/(tn+fp)

    return accuracy, recall, speci , profit, mini, maxi


