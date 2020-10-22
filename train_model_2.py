from utility_function import wavelet_transform, get_X_y_2, autoencoder, init_model, plot_loss
from utility_function import get_prediction_2, get_performance
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import ipdb
import json

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, df, df_test, n_days, length, style):
        self.n_days = n_days
        self.length = length
        self.df = df
        self.df_test = df_test
        self.features = len(df.columns)
        self.style = style

    def train(self):
        df = self.df

        for x in df.columns :
            df[x] = wavelet_transform(df[x])

        self.scaler = MinMaxScaler()
        self.scaler.fit(df)
        df[df.columns]  = self.scaler.transform(df)

        length_of_sequences = [self.length for x in range(820)]
        X_train, y_train = get_X_y_2(df, self.n_days, length_of_sequences, self.style)


        es = EarlyStopping(monitor = 'val_loss',mode = 'min' , verbose = 1, patience = 20, restore_best_weights = True)

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
        self.prediction = get_prediction_2(self.df_test,self.length, 90, self.model, self.scaler)


    def get_perf(self) :
        self.get_prediction()
        prediction = self.prediction
        df_test = self.df_test[self.length:]
        self.perf ,self.hold, self.min , self.max , self.long, self.up = get_performance(df_test, self.prediction)

if __name__ == '__main__':
    n_days = 1
    length = 5
    df  = pd.read_csv('data/data.csv').set_index('date').dropna()
    #ipdb.set_trace()
    n = 820
    s = 0
    return_trim = {}
    return_hold = {}
    detail_perf = {}
    for x in range(31) :
        detail = []
        if x == 0 :
            df_test = df[820-length:910]
            df_ = df[:820]
            t = Trainer(df_, df_test, 1, 5, "linear")
            t.get_perf()
            return_trim[x] = t.perf

            return_hold[x] = t.hold

            detail.append(t.min)
            detail.append(t.max)
            detail.append(t.long)
            detail.append(t.up)
            detail_perf[x] = detail

        else :
            s += 90
            n += 90

            df_test = df[n-length:n+90]
            df_ = df[s:n]
            t = Trainer(df_, df_test, 1, 5,'linear')
            t.get_perf()
            return_trim[x] = t.perf

            return_hold[x] = t.hold

            detail.append(t.min)
            detail.append(t.max)
            detail.append(t.long)
            detail.append(t.up)
            detail_perf[x] = detail


    with open('perf_summary/model_2/performance.json', 'w') as fp:
        json.dump(return_trim, fp,  indent=4)

    with open('perf_summary/model_2/perf_hold.json', 'w') as fp:
        json.dump(return_hold, fp, indent = 4)

    with open('perf_summary/model_2/detail_perf.json', 'w') as fp:
        json.dump(detail_perf, fp, indent = 4)
