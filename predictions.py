# Importing Dependencies

# import tensorflow as tf 
# from tensorflow import keras
# from tensorflow.keras import layers, datasets, models
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.metrics import AUC, CategoricalAccuracy, CategoricalCrossentropy
# import matplotlib.pyplot as plt
# import tensorflow_hub as hub
from PIL import Image
import time
import os
import numpy as np

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# def make_model():
    
#     # input_image = Image.open(r'static/uploads/'+input_image_name)
#     # input_image = input_image.resize((32,32))

#     # Loading the Data 
#     (image_train, label_train), (image_test,label_test) = datasets.cifar10.load_data()

#     image_train, image_test = image_train/255.0, image_test/255.0



#     my_model = Sequential([
#                         Conv2D(64, (3,3), activation='relu', input_shape=(32,32,3)),
#                         MaxPooling2D(pool_size=(2,2)),
#                         Conv2D(64, (3,3), activation='relu'),
#                         MaxPooling2D(pool_size=(2,2)),
#                         Conv2D(32, (3,3), activation='relu'),
#                         MaxPooling2D(pool_size=(2,2)),
#                         Flatten(),
#                         Dense(64, activation='relu'),
#                         Dense(10, activation='softmax')
#     ])

#     early_stopper = EarlyStopping(monitor='val_loss', patience=3)


#     my_model.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
#                 metrics=['accuracy'])

#     history = my_model.fit(image_train, 
#                         label_train, 
#                         epochs=30, 
#                         batch_size=20,
#                         validation_data=(image_test, label_test), 
#                         callbacks=[early_stopper])

#     return my_model

def make_prediction(input_image_name):
    
    input_image = Image.open(r'static/uploads/'+input_image_name)
    input_image = input_image.resize((32,32))


    dir = os.listdir('models/')
    export_path_keras = "./models/prediction_ciphar.h5"

#     if len(dir) == 0:
#         my_model = make_model()
#         my_model.save(export_path_keras)
#         return my_model.predict(input_image)                                            
       
#     else:
    reloaded = tf.keras.models.load_model(
        export_path_keras, 
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    input_image_array = np.asarray(input_image)
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']
    prediction_list = reloaded.predict(input_image_array.reshape(1,32,32,3))
    print(type(prediction_list))

    # prediction = label_names[list(prediction_list).index(max(list(prediction_list)))]
    for i in prediction_list:
        prediction = label_names[list(i).index(max(list(i)))]

    return prediction


