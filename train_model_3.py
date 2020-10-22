from utility_function import wavelet_transform, get_sample,get_X_y, autoencoder, init_model, plot_loss
from utility_function import get_prediction, get_performance_2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import ipdb
import json

class Trainer() :
    def __init__(self, df, df_test, n_days, length, style) :
        self.n_days = n_days
        self.length = length
        self.df = df
        self.df_test = df_test
        self.features = len(df.columns) - 1
        self.style = style


    def train(self):
        df = self.df
        df_ = df.drop(columns = 'up')
        for x in df_.columns :
            df_[x] = wavelet_transform(df_[x])


        self.scaler = MinMaxScaler()
        self.scaler.fit(df_)
        df_[df_.columns]  = self.scaler.transform(df_)


        self.autoencoder, self.encoder = autoencoder(self.features)
        X = np.array(df_)
        X = X.reshape(len(X), 1, self.features)

        es = EarlyStopping(monitor = 'val_loss',mode = 'min' , verbose = 1, patience = 20, restore_best_weights = True)
        self.autoencoder.fit(X,X,
                    validation_split = 0.3,
                   callbacks = [es],
                   epochs = 1000,
                   batch_size = 64,
                   shuffle = True,
                   verbose = False)


        X_encode = self.autoencoder.predict(X)
        X_encode.shape = (X_encode.shape[0], X_encode.shape[2])
        new_df = pd.DataFrame(X_encode)
        new_df['up'] = df['up'].reset_index(drop=True)


        length_of_sequences = [self.length for x in range(820)]
        X_train, y_train = get_X_y(new_df, self.n_days, length_of_sequences, "clf")

        self.model = init_model(self.length, self.n_days, self.features, self.style)

        self.history = self.model.fit(np.array(X_train), np.array(y_train),
                    validation_split = 0.3,
                    callbacks = [es],
                    epochs = 1000,
                    batch_size = 32,
                    shuffle = True,
                    verbose = False)

    def learning_viz(self) :
        self.train
        history = self.history
        plot_loss(history)


    def get_prediction(self) :
        self.train()
        self.prediction = get_prediction(self.df_test,self.length, 90, self.autoencoder, self.model, self.scaler)


    def get_perf(self) :
        self.get_prediction()
        prediction = self.prediction
        df_test = self.df_test[self.length:]
        self.perf ,self.hold, self.min , self.max = get_performance_2(df_test, self.prediction)

    #def predict_viz(self) :


if __name__ == '__main__':
    n_days = 1
    length = 5
    df  = pd.read_csv('data/data.csv').set_index('date').dropna()

    up = [0]
    for i in range(1,len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1] :
            up.append(1)
        else :
            up.append(0)
    df['up'] = up
    #ipdb.set_trace()
    n = 820
    s = 0
    return_trim = {}
    return_hold = {}
    detail_perf = {}
    for x in range(31) :
        detail = []
        if x == 0 :
            df_test = df[820-length:910].drop(columns = 'up')
            df_ = df[:820]
            t = Trainer(df_, df_test, 1, 5, 'clf')
            t.get_perf()
            return_trim[x] = t.perf


            return_hold[x] = t.hold

            detail.append(t.min)
            detail.append(t.max)

            detail_perf[x] = detail

        else :
            s += 90
            n += 90

            df_test = df[n-length:n+90].drop(columns = 'up')
            df_ = df[s:n]
            t = Trainer(df_, df_test, 1, 5, 'clf')
            t.get_perf()
            return_trim[x] = t.perf

            return_hold[x] = t.hold

            detail.append(t.min)
            detail.append(t.max)
            detail_perf[x] = detail


    with open('perf_summary/model_3/performance.json', 'w') as fp:
        json.dump(return_trim, fp,  indent=4)

    with open('perf_summary/model_3/perf_hold.json', 'w') as fp:
        json.dump(return_hold, fp, indent = 4)

    with open('perf_summary/model_3/detail_perf.json', 'w') as fp:
        json.dump(detail_perf, fp, indent = 4)
