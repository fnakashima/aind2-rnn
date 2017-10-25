import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    #print(series)
    # containers for input/output pairs
    X = []
    y = []
    length = len(series)-window_size
    for i in range(length):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    #print("X\n",X)
    #print("y\n",y)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    return model
    pass


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    valid_chars = list(string.ascii_lowercase)
    valid_chars += punctuation
    valid_chars += [' ']

    # find invalid characters in the text
    unique_chars = set(text)
    invalid_chars = []
    for c in unique_chars:
        if c not in valid_chars:
            invalid_chars.append(c)

    # replace the invalid characters with empty spaces
    for c in invalid_chars:
        text = text.replace(c, '')
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    text_list = list(text)
    length = len(text_list)-window_size
    for i in range(0, length, step_size):
        inputs.append(''.join(text_list[i:i + window_size]))
        outputs.append(text_list[i + window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))

    return model
    pass
