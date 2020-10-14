from utility_function import wavelet_transform, get_sample,get_X_y, autoencoder, init_model, plot_loss
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import ipdb

class Trainer() :
    def __init__(self, df, df_test, n_days, length ) :
        self.n_days = n_days
        self.length = length
        self.train_days = 815
        self.predict_days = 90
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
                   shuffle = True)


        X_encode = self.autoencoder.predict(X)
        X_encode.shape = (X_encode.shape[0], X_encode.shape[2])
        new_df = pd.DataFrame(X_encode)
        df.reset_index(inplace = True)
        new_df['price'] = df['price']


        length_of_sequences = [self.length for x in range(3500)]
        X_train, y_train = get_X_y(new_df, self.n_days, length_of_sequences)

        self.model = init_model(self.length, self.n_days)

        self.history = self.model.fit(np.array(X_train), np.array(y_train),
                    validation_split = 0.3,
                    callbacks = [es],
                    epochs = 1000,
                    batch_size = 32,
                    shuffle = True)

    def learning_viz(self) :

        self.train
        history = self.history

        plot_loss(history)


    def get_prediction(self) :
        self.train()
        df_test = self.df_test


        df_test_n = df_test.copy()
        df_test_n['price'] = wavelet_transform(df_test_n)[:len(df_test_n)]

        df_test = df_test[self.length + 1:]

        prediction = []
        for x in range(len(df_test)):
            encode = self.autoencoder.predict(df_test_n[x:x+self.length])
            encode.shape = (1,encode.shape[0], encode.shape[1])
            predict = self.model.predict(encode)
            prediction.append(predict)

        prediction = np.array(prediction)
        prediction.shape = (89)
        self.prediction = prediction
        self.df_test = df_test


    def get_perf(self) :
        self.get_prediction()
        prediction = self.prediction
        df_test = self.df_test
        df_perf = df_test[['price']]
        df_perf['prediction'] = prediction
        df_perf.columns  = ['true', 'prediction']
        long = []
        for i in range(len(df_perf)-1):
            if df_perf['prediction'].iloc[i+1] > df_perf['prediction'].iloc[i] :
                long.append(1)
            else :
                long.append(0)
        long.append(0)
        df_perf['long'] = long


        perf = [0]
        for i in range(len(df_perf)-1):
            if df_perf['long'].iloc[i-1] == 1 :
                x = (df_perf['true'].iloc[i] - df_perf['true'].iloc[i-1]) / df_perf['true'].iloc[i-1]
                perf.append(1 + x)
            else :
                x = (df_perf['true'].iloc[i-1] - df_perf['true'].iloc[i]) / df_perf['true'].iloc[i-1]
                perf.append(1 + x)
        df_perf['perf'] = perf

        self.df_perf = df_perf
        self.perf = df_perf['perf'].iloc[1:].prod()
        print("profit:", self.perf)


if __name__ == '__main__':
    n_days = 1
    length = 5
    df  = pd.read_csv('ma_ema.csv').set_index('date').dropna()
    #ipdb.set_trace()
    n = 820
    s = 0
    for x in range(10000) :
        if x == 0 :
            df_test = df[820-length:910]
            df_ = df[:820]
            t = Trainer(df_, df_test, 1, 5)
            t.get_perf()
        else :
            p =+ 90
            n =+ 90
            if n < len(df):
                df_test = df[n-length:n+90]
                df_ = df[s:n]
                t = Trainer(df_, df_test, 1, 5)
                t.get_perf()
            else :
                break
    #length_of_sequences = [5 for x in range(3500)]
    #a,b = get_X_y(df, 1, length_of_sequences)
    #print(a,b)



