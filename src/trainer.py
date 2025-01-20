import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import argparse
from keras.src.losses import categorical_crossentropy

import models

keras.utils.set_random_seed(42)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('-d', action='store', default=None,
                    dest='data',
                    help='choose sequence file')
parser.add_argument('-gpu', action='store', default="0",
                    dest='gpu',
                    help='choose gpu number')
parser.add_argument('-name', action='store', default="model1",
                    dest='name',
                    help='weights will be stored with this name')
parser.add_argument('-model_name', action='store', default=None,
                    dest='model_name',
                    help='name of the model to call')
parser.add_argument('-log_file', action='store',
                    dest='log_file',
                    help='Log file')

def loss_fn(y_true, y_pred):
        return 1/np.log(2) * categorical_crossentropy(y_true, y_pred)

def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def generate_single_output_data(file_path,batch_size,time_steps):
        series = np.load(file_path)
        series = series.reshape(-1, 1)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        onehot_encoder.fit(series)

        series = series.reshape(-1)

        data = strided_app(series, time_steps+1, 1)
        l = int(len(data)/batch_size) * batch_size

        data = data[:l] 
        X = data[:, :-1]
        Y = data[:, -1:]
        
        Y = onehot_encoder.transform(Y)
        return X,Y

        
def fit_model(X, Y, bs, nb_epoch, model):
        y = Y
        optim = keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=loss_fn, optimizer=optim)
        checkpoint = ModelCheckpoint(arguments.name, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        csv_logger = CSVLogger(arguments.log_file, append=True, separator=';')
        early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.005, patience=3, verbose=1)

        callbacks_list = [checkpoint, csv_logger, early_stopping]
        #callbacks_list = [checkpoint, csv_logger]
        model.fit(X, y, epochs=nb_epoch, batch_size=bs, verbose=1, shuffle=True, callbacks=callbacks_list)



arguments = parser.parse_args()
print(arguments)

batch_size=1200
sequence_length=64
num_epochs=1

X,Y = generate_single_output_data(arguments.data,batch_size, sequence_length)
print(Y.shape[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = getattr(models, arguments.model_name)(batch_size, sequence_length, Y.shape[1])

model.to(device)
X = torch.tensor(data=X, dtype=torch.int32, device=device)
Y = torch.tensor(data=Y, dtype=torch.float, device=device)

fit_model(X, Y, batch_size,num_epochs , model)

