# Ciphar10_Deploy

---------------------------------------------------------------------------------------------------------------------------

#### This model was developed in Python using TensorFlow. The dataset involved was the Ciphar 10 dataset that is listed under tensorflow datasets.

## Idea:
The idea was to integrate a high performing CNN model with a flask user interface where users can upload an image belonging to one of the listed categories and get a
prediction back. The prediction could later be fed into a text-to-speech program depending on the use case. 

## Optimizations
The complexity of neural network is limited to achieving the desired accuracy so the performance during prediction is not affected. 

The trained model has also been exported to .h5 format so that the training is not initiated everytime a user tries to make a prediction. Model code has been commented out. 


## Technologies:
Python
TensorFlow 
Keras
Flask


