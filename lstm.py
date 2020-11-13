from utility_function import wavelet_transform, get_sample,get_X_y, autoencoder, lstm_model, plot_loss
from utility_function import get_prediction_2, get_accuracy_LSTM
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
        self.df_true = df_test.copy()

    def train(self):
        df = self.df
        self.scaler = MinMaxScaler()
        self.scaler.fit(df)
        df[df.columns]  = self.scaler.transform(df)


        X_train, y_train = get_X_y(df, self.n_days, self.length , self.style)

        self.model = lstm_model(self.length, self.n_days, self.features, self.style)

        #es = EarlyStopping(monitor = 'accuracy',mode = 'min' , verbose = 1, patience = 100, restore_best_weights = True)

        self.history = self.model.fit(np.array(X_train), np.array(y_train),
                    #validation_split = 0.3,
                    #callbacks = [es],
                    epochs = 200,
                    batch_size = 64,
                    shuffle = True,
                    verbose = True)

    def learning_viz(self) :
        self.train
        history = self.history
        plot_loss(history)



    def get_perf(self) :
        self.train()
        self.df_true = self.df_true[self.length:]
        self.accuracy , self.recall, self.specificity, self.profit, self.min , self.max = get_accuracy_LSTM(self.df_test, self.df_true,self.model, self.length)






if __name__ == '__main__':

    lenght_list = [5,10,20,50]
    temporal_horizon = [1,7]
    learning_years = ['2y','3y']
    for ye_ in learning_years :
        for te_ in temporal_horizon :
            for le_ in lenght_list :
                for t in range(10) :
                    n_days = 1
                    length = le_
                    df  = pd.read_csv('data/data.csv').set_index('date').dropna()
                    if te_ == 7 :
                        up = [0,0,0,0,0,0,0]
                        for i in range(7,len(df)):
                            if df['close'].iloc[i] > df['close'].iloc[i-7] :
                                up.append(1)
                            else :
                                up.append(0)
                        df['up'] = up
                    if ye_ == '2y' :
                        n = 1790
                    if ye_ == '3y' :
                        n = 1425
                    s = 2520
                    score_trim = {}
                    for x in range(12) :
                        s += 90
                        n += 90
                        score = []
                        df_test = df.iloc[s-length:s+90]
                        df_ = df.iloc[n:s]
                        t = Trainer(df_, df_test, n_days, length, 'clf')
                        t.get_perf()
                        score.append(t.accuracy)
                        score.append(t.recall)
                        score.append(t.specificity)
                        score.append(t.profit)
                        score.append(t.min)
                        score.append(t.max)
                        score_trim[x] = score



                    with open(f'perf_summary/{le_}-1/lstm/{ye_}/score_{t}.json', 'w') as fp:
                        json.dump(score_trim, fp,  indent=4)


