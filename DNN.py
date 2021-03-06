from utility_function import wavelet_transform, get_sample,get_X_y_DNN, autoencoder, DNN_model, plot_loss
from utility_function import get_prediction_2, get_accuracy_DNN
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import ipdb
import json


class Trainer() :
    def __init__(self, df, df_test, n_days, length, style) :
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


    def train(self):
        """
        Scale the Dataframe, chunk our data into X and y samples, Fit the Neural Network
        """
        df = self.df
        self.scaler = MinMaxScaler()
        self.scaler.fit(df)
        df[df.columns]  = self.scaler.transform(df)


        X_train, y_train = get_X_y_DNN(df, self.n_days, self.length , self.style)

        self.model = DNN_model(self.length, self.n_days, self.features)

        #es = EarlyStopping(monitor = 'accuracy',mode = 'min' , verbose = 1, patience = 100, restore_best_weights = True)

        self.history = self.model.fit(np.array(X_train), np.array(y_train),
                    #validation_split = 0.3,
                    #callbacks = [es],
                    epochs = 1000,
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
        self.accuracy , self.recall, self.specificity, self.profit, self.min, self.max= get_accuracy_DNN(self.df_test,self.df_true, self.model, self.length)






if __name__ == '__main__':
    """
    Iterate over the temporal_depth, the temporal_horizon, and the learning years.
    Train and test model for all this parameters over all the quarters.
    Save the results in perf_summary folder
    """
    lenght_list = [5,10,20,50]
    temporal_horizon = [1,7]
    learning_years = ['2y','3y']
    for ye_ in learning_years
        for te_ in temporal_horizon
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



                    with open(f'perf_summary/{le_}-1/DNN/{ye_}/score_{t}.json', 'w') as fp:
                        json.dump(score_trim, fp,  indent=4)

