# Character Convolutions for Text Classification

This is an implmenetation of "Character-level Convolutional Networks for Text Classification" in TensorFlow.

Paper: [https://arxiv.org/pdf/1509.01626.pdf](https://arxiv.org/pdf/1509.01626.pdf)

## Architecture

The architecture of the model is described in section 2.3 of the paper.

### Convolutions with Pooling

There are 6 convolutions with 3 pooling operations.

|  Layer  | ConV Layer Name | Large Feature | Small Feature | Kernel | Pool | Pool Layer Name |
|:-------:|:---------------:|:-------------:|:-------------:|:------:|:----:|:---------------:|
| 1             |  conv_1   |   1024        |   256         | 7      |  3   | pooling_1  |
| 2             | conv_2    |   1024        |   256         | 7      |  3   | pooling_2  |
| 3             |  conv_3   |   1024        |   256         | 3      |  N/A | N/A  |
| 4             | conv_4    |   1024        |   256         | 3      |  N/A | N/A  |
| 5             |  conv_5   |   1024        |   256         | 3      |  N/A | N/A  |
| 6             | conv_6    |   1024        |   256         | 3      |  3   | pooling_3  |

### Dense and Output

|  Layer  | Name | Large Feature | Small Feature |
|:-------:|:---------------:|:-------------:|:-------------:|
| 7       |  dense_out_1   |   2048        |   1024         |
| 8       | dense_out_2    |   2048        |   1024         |
| 9       |  output        |   Number of Classes   |   Number of Classes |

Dropout of probability of 0.5 is included between each fully connected layer.

## Advantages and Disadvantages

### Advantages

* Much faster to train than recurrent-based networks
* The embedding matrix for characters is small compared to the embedding matrix needed for word-level representations.
* Handles out-of-vocabulary words (misspellings, new slang, etc.)

### Disadvantages

* No notion of word level semantics

## Comparison vs LSTM, Universal Sentence Encoder




## Notebook 



