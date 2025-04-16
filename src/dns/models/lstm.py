from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

class LSTMModel:
    def __init__(self, input_shape, output_size):
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=input_shape))
        self.model.add(Dense(output_size))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y, epochs=100, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        return self.model.predict(X)