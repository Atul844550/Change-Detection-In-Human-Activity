🧠 Human Activity Change Detection Using CNN
This project is a deep learning-based solution for detecting and classifying human activities from images using Convolutional Neural Networks (CNNs). It focuses on monitoring changes in human behavior and identifying specific activities for applications in surveillance, health monitoring, and behavior analysis.

The app is deployed with Streamlit to offer a user-friendly interface for testing the model in real time.

🚀 Live Demo
🔗 Streamlit App (Click to Launch)

Replace the # with your Streamlit deployment URL

🧠 Activities Recognized
The model can classify the following 15 human activities:


Class ID	Activity
0	Calling
1	Clapping
2	Cycling
3	Dancing
4	Drinking
5	Eating
6	Fighting
7	Hugging
8	Laughing
9	Listening to Music
10	Running
11	Sitting
12	Sleeping
13	Texting
14	Using Laptop

How It Works
Image Input: Upload a test image through the web app.

Preprocessing: Resize to 128x128, normalize pixel values.

Prediction: The model outputs the predicted activity label.

Display: Shows activity name and confidence score.

🧪 Model Overview
Architecture: Custom CNN with multiple Conv2D, MaxPooling, Flatten, and Dense layers.

Loss Function: Categorical Crossentropy

Optimizer: Adam

Activation: ReLU & Softmax

Evaluation:

Accuracy: ~80%
