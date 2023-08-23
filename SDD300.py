import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Reshape, Concatenate

def build_ssd300(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False)
    
    # Additional convolutional layers for feature extraction
    x = base_model.layers[-1].output
    x = Conv2D(256, (1, 1), activation='relu')(x)
    x = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (1, 1), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (1, 1), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    
    # Prediction layers for detection
    num_prior_boxes = 6
    num_classes_with_bg = num_classes + 1
    
    conf_preds = Conv2D(num_prior_boxes * num_classes_with_bg, (3, 3), padding='same')(x)
    conf_preds = Reshape((-1, num_classes_with_bg))(conf_preds)
    
    loc_preds = Conv2D(num_prior_boxes * 4, (3, 3), padding='same')(x)
    loc_preds = Reshape((-1, 4))(loc_preds)
    
    predictions = Concatenate(axis=-1)([loc_preds, conf_preds])
    
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    return model

input_shape = (300, 300, 3)  # Input image size
num_classes = 20  # Number of classes including background

model = build_ssd300(input_shape, num_classes)
model.summary()