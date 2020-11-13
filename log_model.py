from utility_function import wavelet_transform, get_sample,get_X_y, autoencoder, init_model, plot_loss
from utility_function import get_prediction_2, get_accuracy_LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import ipdb
import json
from sklearn.metrics import accuracy_score, confusion_matrix


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
        X_train = np.array(X_train)
        X_train.shape = (X_train.shape[0], X_train.shape[2])

        self.clf = LogisticRegression().fit(X_train, y_train)

        #es = EarlyStopping(monitor = 'accuracy',mode = 'min' , verbose = 1, patience = 100, restore_best_weights = True)


    def get_perf(self) :
        self.train()

        prediction = self.clf.predict(self.df_test.drop(columns = 'up')[:-1])
        self.accuracy = accuracy_score(df_test['up'][length:].values, prediction)
        tn, fp, fn, tp = confusion_matrix(df_test['up'][length:].values, prediction).ravel()
        self.recall = tp/(tp+fn)
        self.specificity = tn / (tn+fp)


        self.df_true = self.df_true[self.length:]

        profit = 1
        mini = 1
        maxi = 1
        self.df_true['close'] = self.df_true['close'].map(lambda x : np.exp(x))
        for s in range(1,len(self.df_true)):
            if prediction[x-1] == 1 :
                    result = ((self.df_true['close'].iloc[s] -self.df_true['close'].iloc[s-1]) / self.df_true['close'].iloc[s-1]) + 1
                    profit = profit * result
                    if result < mini :
                        mini = result
                    if maxi < result :
                        maxi = result
        self.mini = mini
        self.maxi  = maxi
        self.profit  = profit




if __name__ == '__main__':
    n_days = 1
    length = 1
    df  = pd.read_csv('data_2.csv').set_index('date').dropna()
    #ipdb.set_trace()
    for t in range(2) :
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
                score.append(t.mini)
                score.append(t.maxi)
                score_trim[x] = score

        with open(f'perf_summary/baseline/log_model/{t}.json', 'w') as fp:
                json.dump(score_trim, fp,  indent=4)


