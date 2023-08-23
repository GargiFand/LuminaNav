import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model

def build_crnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Reshaping for the RNN part
    x = Reshape(target_shape=(-1, 128))(x)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Fully connected layer for character prediction
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

input_shape = (128, 32, 1)  # Input image size (height, width, channels)
num_classes = 37  # Number of classes including characters and an "unknown" class

model = build_crnn(input_shape, num_classes)
model.summary()
