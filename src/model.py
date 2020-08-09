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

    Parameters
    ----------
    vocab_size: the number of characters in the vocabulary
    input_length: the size of the input sequences (must be least 160)
    num_classes: the number of output classes
    embedding_size: the vector size of character representations
    model_size: 'large' or 'small' feature sizes

    Returns
    -------
    tf.keras.Model: an uncompiled keras model
    """

    if model_size.lower() == 'small':
        conv_filters = 256
        dnn_size = 1024
    elif model_size.lower() == 'large':
        conv_filters = 1024
        dnn_size = 2048
    else:
        ValueError("model size must be either 'small' or 'large'")

    if input_length < 160:
        ValueError('The input sequences must be at least 160 characters long')

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
                        ) -> Model:
    """Create an LSTM model

    Parameters
    ----------
    vocab_size: the number of characters in the vocabulary
    input_length: the size of the input sequences (must be least 160)
    num_classes: the number of output classes
    embedding_size: the vector size of character representations

    Returns
    -------
    tf.keras.Model: an uncompiled keras model
    """
    model = Sequential()
    model.add(layers.Embedding(
        vocab_size, 
        embedding_size,
        input_length=input_length
    ))
    
    model.add(
        layers.Bidirectional(
            layers.LSTM(
                100,
                dropout=0.15,
                recurrent_dropout=0.15)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def get_use_model(output_size: int,
                  use_url: str=None) -> Model:
    """Create a model based on the Universal Sentence Encoder.
    The model is extracted from TensorFlow Hub.

    See https://arxiv.org/abs/1803.11175.

    Parameters
    ----------
    num_classes: the number of output classes
    use_url: url for the universal sentence encoder model (tfhub)

    Returns
    -------
    tf.keras.Model: an uncompiled keras model
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
