Person-In-Bed Detection Challenge - ICASSP 2025
This repository contains the implementation for the Person-In-Bed Detection Challenge at ICASSP 2025 Signal Processing Grand Challenge. The challenge involves detecting whether a person is in bed using accelerometer data. The project consists of two main tasks:

Task 1: Classification of pre-chunked accelerometer signals into "in bed" or "not in bed."
Task 2: Streaming version of the person-detection classifier that minimizes latency and maximizes accuracy in real-time data.
Directory Structure
Task 1: Classification of Pre-Chunked Accelerometer Signals
In this directory, you'll find the implementation for the classification task, which works on pre-chunked accelerometer data.

train_test_functions.py: Contains the functions for training and testing the classifier.
fourier_head.py: Implements the Fourier Head feature extraction method.
spectral_pooling.py: Implements the Spectral Pooling layer for feature aggregation.
Task 2: Streaming Person Detection
In this directory, the focus is on the real-time streaming detection of a person in bed.

process_training_data.py: Preprocesses the training data for model training.
process_testing_data.py: Preprocesses the testing data for evaluation.
train_model.py: Contains the training function for the streaming model.
smooth_results.py: A function that smooths the model predictions to minimize fluctuations.
generate_submission_csv.py: Generates the final CSV file for the submission.
Installation
To get started with this project, clone this repository and install the necessary dependencies:

bash
复制代码
git clone https://github.com/yourusername/person-in-bed-detection.git
cd person-in-bed-detection
pip install -r requirements.txt
Usage
Task 1: Classification
To train and test the classification model on pre-chunked accelerometer data, run the following command:

bash
复制代码
python task1/train_test_functions.py
You can modify the hyperparameters or the feature extraction methods (Fourier Head, Spectral Pooling) as needed in the corresponding files.

Task 2: Streaming Detection
For training the streaming detection model:

Preprocess the training and testing data:

bash
复制代码
python task2/process_training_data.py
python task2/process_testing_data.py
Train the model:

bash
复制代码
python task2/train_model.py
Smooth the results:

bash
复制代码
python task2/smooth_results.py
Generate the final submission CSV:

bash
复制代码
python task2/generate_submission_csv.py
Results
This project was designed to maximize accuracy while minimizing latency for both the classification and streaming tasks. For Task 1, the goal was to classify the accelerometer data correctly into "in bed" or "not in bed." Task 2 aimed to detect the person in real-time with minimal latency and accurate predictions.

Contributing
Feel free to fork this repository, submit issues, and contribute to the project. Pull requests are always welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

