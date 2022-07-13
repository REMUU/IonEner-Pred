# Import Libraries
from sklearn.preprocessing import (normalize, power_transform, binarize, maxabs_scale, minmax_scale, quantile_transform, StandardScaler)
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (Callback, EarlyStopping)
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam, RMSprop,SGD
import nni
import logging
import pandas as pd
import numpy as np

# Define a NN by TF2 class
class NN(Model):
    def __init__(self, hidden_size, drop_rate, data_shape):
        super().__init__()
        self.fc1 = Dense(hidden_size, input_shape= data_shape,activation='relu')
        self.batchnorm1 = BatchNormalization()
        self.dropout1 = Dropout(drop_rate)
        self.fc2 = Dense(hidden_size, activation='relu')
        self.batchnorm2 = BatchNormalization()
        self.dropout2 = Dropout(drop_rate)
        self.fc3 = Dense(hidden_size, activation='relu')
        self.batchnorm3 = BatchNormalization()
        self.dropout3 = Dropout(drop_rate)
        self.outputlayer = Dense(1)
    def call(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)
        return self.outputlayer(x)

# Load Data
data_regularization = ['l2', 'power_transform', 'binarize', 'maxabs_scale', 'minmax_scale', 'quantile_transform', 'StandardScale']
def prepare_data(X, prepro):
    if prepro == 'l2':
        return normalize(X, norm='l2')
    if prepro == 'power_transform':
        return power_transform(X)
    if prepro == 'binarize':
        return binarize(X)
    if prepro == 'maxabs_scale':
        return maxabs_scale(X)
    if prepro == 'minmax_scale':
        return minmax_scale(X)
    if prepro == 'quantile_transform':
        return quantile_transform(X)
    if prepro == 'StandardScale':
        stdsc = StandardScaler()
        stdsc.fit(X)
        return stdsc.transform(X)


def split_data(data, train_size, prepro):
    Train, Test = train_test_split(data, train_size=train_size, random_state=1001)
    X_train, y_train = Train.iloc[:, 5:], Train['IE']
    X_test, y_test = Test.iloc[:, 5:], Test['IE']
    # data preprocessing
    X_train, X_test = prepare_data(X_train, prepro), prepare_data(X_test, prepro)
    return (X_train, y_train), (X_test, y_test)
    
# Define Experiment Settings
class ReportIntermediates(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_loss'])
        else:
            nni.report_intermediate_result(logs['val_mae'])

# create logs
_logger = logging.getLogger(r'output_log_path')
_logger.setLevel(logging.INFO)

# define rmse loss function
def rmse_loss(y_test, y_pred):
    rmse_loss = tf.sqrt(tf.reduce_mean(tf.square(y_test - y_pred)))
    return rmse_loss


def main(params):
    # load data
    _csv = r'input_csv_path'
    data = pd.read_csv(_csv)
    (X_train, y_train), (X_test, y_test) = split_data(
        data, 0.8, 'quantile_transform'
    )
    data_shape = [len(data.iloc[:, 5:].columns)]
    model = NN(
        hidden_size=params['hidden size'],
        drop_rate=params['dropout rate'],
        data_shape=data_shape
    )
    if params['optimizer'] == 'Adam':
        optimizer = Adam(lr=params['learning rate'])
    elif params['optimizer'] == 'SGD':
        optimizer = SGD(lr=params['learning rate'])
    elif params['optimizer'] == 'RMSprop':
        optimizer = RMSprop(lr=params['learning rate'])
    model.compile(
        optimizer=optimizer,
        #loss=rmse_loss,
        loss='mse',
        metrics=['mae', 'mse'])
    _logger.info('Model built')
    early_stop = EarlyStopping(monitor='val_loss', patience=50)
    model.fit(
        X_train,
        y_train,
        batch_size=params['batch size'],
        epochs=1000,
        validation_split=0.1,
        verbose=0,
        callbacks=[ReportIntermediates(), early_stop]
    )
    _logger.info('Training completed')
    loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
    nni.report_final_result(np.sqrt(mse))  # send rmse to NNI tuner and web UI
    _logger.info('Final accuracy reported: %s', np.sqrt(mse))


if __name__ == '__main__':
    params = {
        'dropout rate': 0.5,
        'hidden size': 256,
        'batch size': 16,
        'learning rate': 1e-4,
        'optimizer': 'Adam'
    }
    tuned_params = nni.get_next_parameter()
    params.update(tuned_params)
    _logger.info('Hyper-parameters: %s', params)
    main(params)