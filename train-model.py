from utility_function import wavelet_transform, get_sample,get_X_y, autoencoder, init_model, plot_loss
from utility_function import get_prediction, get_performance
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import ipdb
import json

class Trainer() :
    def __init__(self, df, df_test, n_days, length ) :
        self.n_days = n_days
        self.length = length
        self.df = df
        self.df_test = df_test


    def train(self):
        df = self.df

        self.autoencoder, self.encoder = autoencoder(14)
        X = np.array(df)
        X = X.reshape(len(X), 1, 14)

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
        df.reset_index(inplace = True)
        new_df['price'] = df['price']


        length_of_sequences = [self.length for x in range(820)]
        X_train, y_train = get_X_y(new_df, self.n_days, length_of_sequences)

        self.model = init_model(self.length, self.n_days)

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
        self.prediction = get_prediction(self.df_test,self.length, 90, self.autoencoder, self.model)


    def get_perf(self) :
        self.get_prediction()
        prediction = self.prediction
        df_test = self.df_test[self.length:]
        self.perf ,self.hold, self.min , self.max , self.long, self.up = get_performance(df_test, self.prediction)

    #def predict_viz(self) :


if __name__ == '__main__':
    n_days = 1
    length = 5
    df  = pd.read_csv('ma_ema.csv').set_index('date').dropna()
    #ipdb.set_trace()
    n = 820
    s = 0
    return_trim = {}
    return_hold = {}
    detail_perf = {}
    for x in range(30) :
        detail = []
        if x == 0 :
            df_test = df[820-length:910]
            df_ = df[:820]
            t = Trainer(df_, df_test, 1, 5)
            t.get_perf()
            return_trim[x] = t.perf
            print(f"profit of period {x}: {t.perf}")

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
            t = Trainer(df_, df_test, 1, 5)
            t.get_perf()
            return_trim['x'] = t.perf

            return_hold[x] = t.hold

            detail.append(t.min)
            detail.append(t.max)
            detail.append(t.long)
            detail.append(t.up)
            detail_perf[x] = detail


    with open('performance.json', 'w') as fp:
        json.dump(return_trim, fp,  indent=4)

    with open('perf_hold.json', 'w') as fp:
        json.dump(return_hold, fp, indent = 4)

    with open('detail_perf.json', 'w') as fp:
        json.dump(detail_perf, fp, indent = 4)




