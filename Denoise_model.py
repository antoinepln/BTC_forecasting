from utility_function import wavelet_transform, get_sample,get_X_y, autoencoder, lstm_model, plot_loss
from utility_function import get_prediction_2, get_accuracy_LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import ipdb
import json


class Trainer() :
    def __init__(self, df, df_test, n_days, length, style, model) :
        """
        instantiate trainer object with the training Dataframe (df), the testing
        Dataframe (df_test), the temporal_horizon n_days, the tremporal depth length,
        the style of the network to train 'clf' or 'regressor'.
        """
        self.n_days = n_days
        self.length = length
        self.df = df
        self.df_test = df_test
        self.features = len(df.columns) - 1
        self.style = style
        self.df_true = df_test.copy()
        self.model = model

    def train(self):
        """
        Scale the Dataframe, depends on the model, encode the features, Wavelet
        Transformation, quadratic discriminant analysis, chunk our data into X and
        y samples, Fit the Neural Network.
        """
        df = self.df
        self.scaler = MinMaxScaler()


        if self.model == 'encoder' :

            self.scaler.fit(df)
            df[df.columns]  = self.scaler.transform(df)

            self.autoencoder = autoencoder(self.features)

            X = np.array(self.df.drop(columns = 'up'))
            X = X.reshape(len(X), 1, self.features)

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
            new_df['up'] = df['up']

            X_train, y_train = get_X_y(new_df, self.n_days, self.length , self.style)

        elif self.model == 'wav' :

            for col_ in df.drop(columns = 'up').columns :
                df[col_] = wavelet_transform(df[col_])[:len(df)]

            self.scaler.fit(df)
            df[df.columns]  = self.scaler.transform(df)

            X_train, y_train = get_X_y(df, self.n_days, self.length , self.style)

        elif self.model = 'qda' :

            self.scaler.fit(df)
            df[df.columns]  = self.scaler.transform(df)

            X_train, y_train = get_X_y(df, self.n_days, self.length , self.style)

            clf = QuadraticDiscriminantAnalysis()

            train = []
            for s in range(len(X_train)) :
                train.append(X_train[s].ravel())
            X_train = train

            clf.fit(X_train, y_train)

            train = []
            for s in range(len(X_train)) :
                quad_dis = clf.predict(X_train[s].reshape(X_train[s].shape[0],1))
                train.append(quad_dis.reshape(self.length, self.features))

            X_train = train

        elif self.model == 'wav_encoder' :
            for col_ in df.drop(columns = 'up').columns :
            df[col_] = wavelet_transform(df[col_])[:len(df)]

            self.scaler = MinMaxScaler()
            self.scaler.fit(df)
            df[df.columns]  = self.scaler.transform(df)

            self.autoencoder = autoencoder(self.features)

            X = np.array(self.df.drop(columns = 'up'))
            X = X.reshape(len(X), 1, self.features)

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
            new_df['up'] = df['up']

            X_train, y_train = get_X_y(new_df, self.n_days, self.length , self.style)

        else :
            for col_ in df.drop(columns = 'up').columns :
                df[col_] = wavelet_transform(df[col_])[:len(df)]

            self.scaler = MinMaxScaler()
            self.scaler.fit(df)
            df[df.columns]  = self.scaler.transform(df)

            X_train, y_train = get_X_y(df, self.n_days, self.length , self.style)

            clf = QuadraticDiscriminantAnalysis()

            train = []
            for s in range(len(X_train)) :
                train.append(X_train[s].ravel())
            X_train = train

            clf.fit(X_train, y_train)

            train = []
            for s in range(len(X_train)) :
                quad_dis = clf.predict(X_train[s].reshape(X_train[s].shape[0],1))
                train.append(quad_dis.reshape(self.length, self.features))

            X_train = train



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
        """
        Get the metrics of our train Nural Network
        """
        self.train()
        self.df_true = self.df_true[self.length:]
        self.accuracy , self.recall, self.specificity, self.profit, self.min , self.max = get_accuracy_LSTM(self.df_test, self.df_true,self.model, self.length)






if __name__ == '__main__':
    """
    Iterate over the models.
    Train and test model for all this models over all the quarters.
    Save the results in perf_summary folder
    """

    model_list = ['wav','encoder','qda','wav_encoder','wav_qda']
    for mod_ in model_list :
        for t in range(10) :
            n_days = 1
            length = 10
            df  = pd.read_csv('data/data.csv').set_index('date').dropna()
            n = 1425
            s = 2520
            score_trim = {}
            for x in range(12) :
                s += 90
                n += 90
                score = []
                df_test = df.iloc[s-length:s+90]
                df_ = df.iloc[n:s]
                t = Trainer(df_, df_test, n_days, length, 'clf',mod_)
                t.get_perf()
                score.append(t.accuracy)
                score.append(t.recall)
                score.append(t.specificity)
                score.append(t.profit)
                score.append(t.min)
                score.append(t.max)
                score_trim[x] = score



            with open(f'perf_summary/lstm_{mod_}/score_{t}.json', 'w') as fp:
                json.dump(score_trim, fp,  indent=4)


