Title: Human Activity Change Detection Using CNN

Overview:

This project is focused on detecting and classifying human activities using images. A Convolutional Neural Network (CNN) model was built to identify 15 different types of human behaviors. The application is deployed using Streamlit, allowing users to upload images and receive real-time activity predictions with a confidence score.

Live Demo:

You can access the deployed Streamlit app from the following link: [Insert your Streamlit link here].

Activities Detected:

The model is trained to classify the following 15 activities:

Calling

Clapping

Cycling

Dancing

Drinking

Eating

Fighting

Hugging

Laughing

Listening to Music

Running

Sitting

Sleeping

Texting

Using Laptop

How It Works:

The user uploads an image through the Streamlit interface.

The image is preprocessed (resized to 128x128 pixels and normalized).

The trained CNN model predicts the most likely activity.

The predicted label and confidence score are displayed to the user.

Model Details:

The model uses multiple convolutional and pooling layers followed by dense layers.

It is trained with categorical crossentropy as the loss function and Adam as the optimizer.

It uses ReLU activation in hidden layers and softmax in the output layer.

The model achieved around [insert accuracy]% accuracy during evaluation.

Dataset Format:

The training data is organized into folders, where each folder corresponds to an activity (e.g., train/calling, train/sitting, etc.). The test folder contains images without labels for prediction purposes.

How to Use:

Clone the repository.

Install required dependencies.

Run the Streamlit app.

Upload an image and get activity prediction.

Dependencies:

The project uses the following major libraries:

TensorFlow

Keras

Streamlit

OpenCV

NumPy

Matplotlib

Pandas

Project Structure:

Jupyter Notebook for training the CNN model.

Streamlit script for deployment (streamlit_app.py).

Saved model (best_model.h5).

train/ and test/ folders containing images.

requirements.txt file for dependencies.
