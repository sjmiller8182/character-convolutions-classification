"""Model creation
"""

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
import tensorflow_hub as hub

DEFAULT_USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

def get_convolutional_model(vocab_size: int,
                            input_length: int,
                            num_classes: int,
                            embedding_size: int=300,
                            model_size: str='small'
                            ) -> Model:
    """Create a character convolutional model
    """

    if model_size.lower() == 'small':
        conv_filters = 256
        dnn_size = 1024
    elif model_size.lower() == 'large':
        conv_filters = 1024
        dnn_size = 2048
    else:
        ValueError("model size must be either 'small' or 'large'")

    model = Sequential()
    model.add(layers.Embedding(
        vocab_size,
        embedding_size,
        input_length=input_length,
        name='character_embedding'
    ))

    model.add(layers.Dropout(0.2, name='input_dropout'))

    model.add(layers.Conv1D(
        filters=conv_filters, 
        kernel_size=7, 
        activation='relu',
        name='conv_1'))
    model.add(layers.MaxPooling1D(
        pool_size=3,
        name='pooling_1'))
    model.add(layers.Conv1D(
        filters=conv_filters, 
        kernel_size=7, 
        activation='relu',
        name='conv_2'))
    model.add(layers.MaxPooling1D(
        pool_size=3,
        name='pooling_2'))

    model.add(layers.Conv1D(
        filters=conv_filters, 
        kernel_size=3, 
        activation='relu',
        name='conv_3'))
    model.add(layers.Conv1D(
        filters=conv_filters, 
        kernel_size=3, 
        activation='relu',
        name='conv_4'))
    model.add(layers.Conv1D(
        filters=conv_filters, 
        kernel_size=3, 
        activation='relu',
        name='conv_5'))

    model.add(layers.Conv1D(
        filters=conv_filters,
        kernel_size=7, 
        activation='relu',
        name='conv_6'))
    model.add(layers.MaxPooling1D(
        pool_size=3,
        name='pooling_3'))
    model.add(layers.Flatten(name='flatten'))

    model.add(layers.Dense(dnn_size,
                           activation='relu',
                           name='dense_out_1'))
    model.add(layers.Dropout(0.5, name='post_dropout_1'))

    model.add(layers.Dense(dnn_size,
                           activation='relu',
                           name='dense_out_2'))
    model.add(layers.Dropout(0.5, name='post_dropout_2'))

    model.add(layers.Dense(num_classes,
                           activation='softmax',
                           name='output'))

    return model

def get_recurrent_model(vocab_size: int,
                        input_length: int,
                        num_classes: int,
                        embedding_size: int=300
                        ):
    """Create an LSTM model
    """
    model = Sequential()
    model.add(layers.Embedding(
        vocab_size, 
        embedding_size,
        input_length=input_length
    ))
    
    model.add(
        layers.LSTM(
            100,
            dropout=0.15,
            recurrent_dropout=0.15))

    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def get_use_model(output_size: int,
                  use_url: str=None):
    """Create a model based on the Universal Sentence Encoder.
    The model is extracted from TensorFlow Hub.

    See https://arxiv.org/abs/1803.11175.
    """

    if use_url is None:
        use_url = DEFAULT_USE_URL

    use = hub.KerasLayer(use_url,
                     trainable=False,
                     input_shape=[], 
                     dtype=tf.string)

    model = Sequential()
    model.add(use)
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_size, activation='softmax'))

    return model
