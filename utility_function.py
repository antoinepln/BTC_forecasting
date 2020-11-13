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
    '''generates a sample X, the number of rows of the vector X equals the length,
     and the number of columns equals the temporal_horizon. generates the single
     target y associated with the sample X. Takes as input the dataframe from
     which the samples are extracted,length, temporal_horizon ,the style of the
     model from which the data will be entered, 'clf' or 'regressor', Start is
     the location of the first line of the sample X.'''

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
    """Takes as input the dataframe from which the samples are extracted ,we
    iterate on it to generate all the pairs of X_sample and y_sample available
    in our DataFrame. the function takes the length, the temporal_horizon of the samples, and
    the style of the model. """

    X, y = [], []

    for len_ in range(len(df)-temporal_horizon -length):
        xi, yi = get_sample(df, length, temporal_horizon, style , len_)
        X.append(xi)
        y.append(yi)

    return X, y



def get_X_y_DNN(df, temporal_horizon, length, style):
    """same as get_X_y, except that we transform our X vector into a vector with
    a dimension equal to temporal_horizon * length, in order to pass it to our DNN
    network. """
    X, y = [], []

    for len_ in range(len(df)-temporal_horizon -length):
        xi, yi = get_sample(df, length, temporal_horizon, style , len_)
        X.append(xi.ravel())
        y.append(yi.ravel())

    return X, y

def wavelet_transform(df):
    """ we take as input the column of the DataFrame to be transformed, we
    decompose it then reconstruct the signal, and return a new 'de-noise' series. """
    ca , cb = pywt.wavedec(df.values, 'haar', level = 1)
    cat = pywt.threshold(ca, np.std(ca), mode = 'soft')
    cbt = pywt.threshold(cb, np.std(cb), mode = 'soft')
    coeff = [cat , cbt ]
    return pywt.waverec(coeff, 'haar')

def autoencoder(features):
    """we take as input the number of features to be encoded, and return the neural
     network autoencoder"""
    input_data = Input(shape=(1, features))
    encoded1 = Dense(features, activation="relu", activity_regularizer=regularizers.l2(0))(input_data)
    one_l = Dense(1, activation="relu", activity_regularizer=regularizers.l2(0))(encoded1)
    decoded = Dense(features, activation="linear", activity_regularizer=regularizers.l2(0))(one_l)
    autoencoder = Model(inputs=input_data, outputs=decoded)
    encoder = Model(input_data, one_l)
    autoencoder.compile(loss = 'mse', optimizer = 'rmsprop',metrics = ['mae'])
    return autoencoder


def lstm_model(length, n_days, features, style) :
    """ We take as input the characteristics of the network, temporal depth = length,
     temporal_horizon = n_days, number of fetaures = features, style = 'clf' or
     'regressor', the function returns the model to us"""
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
    """ We take as input the characteristics of the network, temporal depth = length,
     temporal_horizon = n_days, number of fetaures = features, the function returns
    the model to us"""
    nb = length * features
    model = Sequential()
    model.add(Dense(nb,activation = 'relu',input_dim=(nb)))
    model.add(Dense(nb/2,activation = 'relu'))
    model.add(Dense(nb/4,activation = 'relu'))
    model.add(Dense(1,activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def plot_loss(history):
    """ Takes the fitting history of the network and returns the learning curves """

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()




def get_accuracy_LSTM(df_test, df_true, model, length) :
    """ Takes the test portion of the transformed and scaled DataFrame to perform
    prediction (df_test), and the test portion without logarithmic transformation
    or sclaing to evaluate performance (df_true), trained model, time depth (length).
    Returns accuracy score, recall, specificity, yield, maximum loss in one day, and
     maximum gain in one day """
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
    """ same as the previous function but adapted to DNN networks """
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


