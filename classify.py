# lstm autoencoder recreate sequence
from argparse import ArgumentParser
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import EarlyStopping
import datetime
import os
import logging
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
import tensorflow as tf
import sys
import pickle
import math
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()

now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

NEURON_N_DEFAULT = 100
N_EPOCH_DEFAULT = 100       

def import_training_data(training_folder_kn, training_folder_no_kn, kn_frac=0.5):
    """
    Imports the spectra used for training, as well
    as the labels. kn_frac is fraction of training
    samples with KN spectra embedded.
    """
    
    
def make_model(inputN, hiddenN, neuronN):
    """
    Make simple MLP for spectrum classification.

    Parameters
    ----------
    inputN : int
        Number of parameters in input layer.
    hiddenN : int
        Number of hidden layers in MLP.
    neuronN : int
        Number of neurons per hidden layer.
        
    Returns
    -------
    model : keras.models.Model
        Moel to be trained
    callbacks_list : list
        List of keras callbacks
    """
    assert hiddenN >= 1
    
    model = Sequential()
    model.add(Dense(neuronN, input_shape=(inputN,)))
    for i in range(hiddenN - 1):
        model.add(Dense(neuronN))
    
    model.add(Dense(2))
    
    new_optimizer = Adam(learning_rate=1e-4)
    
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
                       verbose=0, mode='min', baseline=None,
                       restore_best_weights=True)

    callbacks_list = [es]

    model.compile(optimizer=new_optimizer, loss=BinaryCrossentropy(from_logits=True), metrics=[BinaryAccuracy(),])
    return model, callbacks_list


def fit_model(model, callbacks_list, X, y_true, n_epoch, history_dict="./trainHistoryDict"):
    """
    Fit our model to spectra.

    Parameters
    ----------
    model : keras.models.Model
        The model to train
    callbacks_list : list
        List of keras callbacks
    X : numpy.ndarray
        input spectra
    y_true : numpy.ndarray
        true labels of data (0 = no kn, 1 = kn)
    n_epoch : int
        Number of epochs to train for
    history_dict : str
        Where to save the training history

    Returns
    -------
    model : keras.models.Model
        Trained keras model
    """
    history = model.fit(X, y_true
                        batch_size=32,
                        epochs=n_epoch,
                        verbose=1,
                        shuffle=False,
                        callbacks=callbacks_list,
                        validation_split=0.1,
                       )
                        
    with open(history_dict, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
    return model


def save_model(model, hiddenN, neuronN, model_dir='models/', outdir='./'):
    # make output dir
    model_dir = outdir + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_dir+"model_"+date+"_"+str(hiddenN)+'_'+str(neuronN)+".json", "w") as json_file:
        json_file.write(model_json)
    with open(model_dir+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_dir+"model_"+date+"_"+str(hiddenN)+'_'+str(neuronN)+".h5")
    model.save_weights(model_dir+"model.h5")

    logging.info(f'Saved model to {model_dir}')

    
def main():
    parser = ArgumentParser()
    parser.add_argument('training_folder', type=str, help='Folder with training data')
    parser.add_argument('--outdir', type=str, default='./products/',
                        help='Path in which to save the model data')
    parser.add_argument('--neuronN', type=int, default=NEURON_N_DEFAULT, help='Number of neurons in hidden layers')
    parser.add_argument('--hiddenN', type=int, default=ENCODING_N_DEFAULT,
                        help='Number of neurons in encoding layer')
    parser.add_argument('--n-epoch', type=int, dest='n_epoch',
                        default=N_EPOCH_DEFAULT,
                        help='Number of epochs to train for')

    args = parser.parse_args()
    
    X, y_true = import_training_data(args.training_folder)

    inputN = len(X[0])

    model, callbacks_list = make_model(inputN, args.hiddenN, args.neuronN)
    model = fit_model(model, callbacks_list, X, y_true, args.n_epoch)

    if args.outdir[-1] != '/':
        args.outdir += '/'
        
    save_model(model, args.hiddenN, args.neuronN, outdir=args.outdir)
        
if __name__ == '__main__':
    main()
