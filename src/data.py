"""Data handling
"""

from collections import defaultdict, Counter
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_data():
    """Read the csv and return X_train, X_test, y_train, y_test
    """
    names = ['polarity','NA1','NA2','NA3','NA4', 'text']
    train_data = pd.read_csv(
        './training.1600000.processed.noemoticon.csv',
        encoding='latin1', 
        names=names
    )
    # shuffle the data
    train_data = train_data.sample(frac=1.0, random_state=42)
    # split of a test set
    split_loc = int(train_data.shape[0] * 0.8) # 80/20 split
    test_data = train_data.iloc[split_loc:]
    train_data = train_data.iloc[:split_loc]

    # convert all strings to lower case
    train_data.text = train_data.text.str.lower()
    test_data.text = test_data.text.str.lower()

    # remove any cases where there is more than one space
    # since I am tokenizing on ' '
    multi_space_pattern = r'[\s]{2,}'
    train_data.text = train_data.text.str.replace(multi_space_pattern, ' ')
    test_data.text = test_data.text.str.replace(multi_space_pattern, ' ')

    # convert the sentiment to encoding
    polarity_encoder = LabelEncoder()
    train_data['y'] = polarity_encoder.fit_transform(train_data.polarity)
    test_data['y'] = polarity_encoder.transform(test_data.polarity)

    X_train = train_data.text.values
    X_test = test_data.text.values
    y_train = to_categorical(train_data.y.values)
    y_test = to_categorical(test_data.y.values)
    
    return X_train, X_test, y_train, y_test

class CharacterTranslator:
    """Translate between strings and character-tokenized numeric-encoded sequences
    """
    def __init__(self, unigram: bool = True):
        """Constructor
        """
        # whether to use unigram (True) or 
        # bigram (False) tokenization
        self.unigram = unigram
        # special tokens
        self.start_token = 1
        self.pad_token = 0
        self.oov_token = 2
        # special strings
        self.start_str = '<start>'
        self.pad_str = '<pad>'
        self.oov_str = '<OOV>'
        # dictionaries
        self.str_to_token = defaultdict(self._get_oov_token)
        self.token_to_str = defaultdict(self._get_oov_str)
        # set special tokens
        for token, word in zip([self.start_token,
                                self.pad_token,
                                self.oov_token],
                               [self.start_str,
                                self.pad_str,
                                self.oov_str]):
            self.str_to_token[word] = token 
            self.token_to_str[token] = word
    
        # collect the length distribution
        self.seq_sizes = None
    
    def _get_oov_token(self) -> int:
        """Return default for defaultdict
        """
        return self.oov_token
    
    def _get_oov_str(self) -> str:
        """Return default for defaultdict
        """
        return self.oov_str
    
    @property
    def vocab_size(self):
        """The size of the vocabulary from fitting
        """
        return len(self.str_to_token.keys())

    @staticmethod
    def _tokenize(sequences, unigram):
        """Tokenize sequences (unigram or bigram)
        """
        tokenized = list()
        if unigram:
            for string in sequences:
                tokenized.append(list(string))
        else:
            for string in sequences:
                # tokenize by taking two elements at a time
                tokenized.append(
                    [string[i:i+2] for i in range(0, len(string), 2)]
                )
        
        return tokenized
    
    def _encode(self,
                tokenized_seqs):
        """Encode tokenized sequences
        """
        encoded = list()
        for seq in tokenized_seqs:
            encoded_seq = list()
            for element in seq:
                encoded_seq.append(self.str_to_token[element])
            encoded.append(encoded_seq)
        return encoded
    
    def fit(self,
            sequences: List[str],
            vocab_limit: int = None) -> None:
        """Build the dictionaries from the strings

        Parameters
        ----------
        sequences: a List of strings to fit the translator
        vocab_limit: a threshold to limit the tokens to the most common; 
            others are mapped to (<OOV>).
        """
        # tokenize
        tokenized = self._tokenize(sequences, self.unigram)
        
        # calculate the sizes of sequences
        sizes = list()
        for seq in tokenized:
            sizes.append(len(seq))
        self.seq_sizes = np.array(sizes)
        
        # build the encodings
        counts = Counter([x for xs in tokenized for x in xs])
        tokens = [k for k, v in counts.most_common(vocab_limit)]
        
        codes = np.arange(3, len(tokens) + 3)
        for token, code in zip(tokens, codes):
            self.str_to_token[token] = code 
            self.token_to_str[code] = token
    
    def transform(self,
                  sequences: List[str],
                  maxlen: int):
        """Transform a list of sentences to numeric sequences

        Parameters
        ----------
        sequences: a List of strings to tokenize and encode
        maxlen: the new size of all sequences
        """
        # tokenize
        tokenized = self._tokenize(sequences, self.unigram)
        encoded = self._encode(tokenized)
        transformed = pad_sequences(encoded, maxlen=maxlen)
        
        return transformed
    
    def fit_transform(self,
                      sequences: List[str],
                      maxlen: int,
                      vocab_limit: int = None
                      ):
        """Fit on given dataset, then transform given dataset

        Parameters
        ----------
        sequences: a List of strings to tokenize and encode
        maxlen: the new size of all sequences
        vocab_limit: a threshold to limit the tokens to the most common; 
            others are mapped to (<OOV>).
        """
        self.fit(sequences, vocab_limit)
        return self.transform(sequences, maxlen)

    def inverse_transform(self,
                          sequences: List[str],
                          unigram: bool=True):
        """Convert a set of numeric sequences back to sentences.
        This is lossy because of out-of-vocabulary words are lost in the transform operation.
        """
        decoded = list()
        for seq in sequences:
            decoded_seq = list()
            for element in seq:
                decoded_seq.append(self.token_to_str[element])
            decoded.append(''.join(decoded_seq))
        return decoded
