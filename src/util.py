"""Data utilities
"""

import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_distance

def similarity(vector1: np.ndarray,
               vector2: np.ndarray) -> float:
               """Calculate cosine similarity between two vectors
               """
               return 1 - cosine_distance(vector1, vector2)

def plot_loss_accuracy(history):
    """Plot loss and accuracy history
    """
    
    # find the minimum loss epoch
    minimum = np.min(history.history['val_loss'])
    min_loc = np.where(minimum == history.history['val_loss'])[0]
    # get the vline y-min and y-max
    loss_min, loss_max = (min(history.history['val_loss'] + history.history['loss']),
                          max(history.history['val_loss'] + history.history['loss']))
    acc_min, acc_max = (min(history.history['val_accuracy'] + history.history['accuracy']),
                        max(history.history['val_accuracy'] + history.history['accuracy']))
    
    # create the plots
    fig, ax = plt.subplots(ncols=2, figsize = (15,7))
    index = np.arange(1, len(history.history['accuracy']) + 1)

    # plot the loss history
    ax[0].plot(index, history.history['loss'], label = 'loss')
    ax[0].plot(index, history.history['val_loss'], label = 'val_loss')
    ax[0].vlines(min_loc + 1, loss_min, loss_max, label = 'min_loss_location')
    ax[0].set_title('Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()

    # plot the accuracy history
    ax[1].plot(index, history.history['accuracy'], label = 'accuracy')
    ax[1].plot(index, history.history['val_accuracy'], label = 'val_accuracy')
    ax[1].vlines(min_loc + 1, acc_min, acc_max, label = 'min_loss_location')
    ax[1].set_title('Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()

    plt.show();
