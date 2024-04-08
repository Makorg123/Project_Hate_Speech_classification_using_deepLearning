# Project: Hate Speech Classification with LSTM RNN

### Overview
This repository contains the code and resources for a hate speech classification project using a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN). The goal of this project is to classify text data into categories of hate speech, offensive language, and normal speech.

### Dataset
The dataset used for this project consists of text data related to hate speech, obtained from kaggle. Due to the imbalance in the original dataset, it was merged with additional data to create a more balanced dataset for training the model.

### Preprocessing
- Conducted Exploratory Data Analysis (EDA) to understand the underlying patterns in the text data.
- Employed text preprocessing techniques such as:
- Stemming
- Stopwords removal
- Cleaning of text by eliminating tags and hyperlinks

### Model Architecture
- Implemented a LSTM (Long Short-Term Memory) model using Keras with TensorFlow backend.
- Utilized the LSTM model for effective sequence modeling, ideal for capturing patterns in text data.
- Model layers include:
- LSTM
- Activation
- Dense
- Dropout
- Embedding
  
### Training and Evaluation
- Split the dataset into training and testing sets.
- Tokenized the text data using a Tokenizer and applied pad_sequences for uniform sequence lengths.
- Optimized the model using the RMSprop optimizer and utilized a sigmoid activation function for binary classification.
- Evaluated model performance using a confusion matrix to analyze hate speech, offensive language, and normal speech classification.

### Usage
1) Clone the repo:
`git clone https://github.com/your-username/hate-speech-classification-lstm.git`

2) Install the required libraries:
`pip install -r requirements.txt`

3) Run the Jupyter Notebook:
`jupyter notebook hate_speech_classification.ipynb`

### Requirements
- Python 3.x
- TensorFlow
- Keras
- Pandas
- Numpy
- Matplotlib
- Plotly
- Scikit-learn

### Results
The LSTM RNN model demonstrates strong performance in classifying hate speech, offensive language, and normal speech categories. The provided Jupyter Notebook (hate_speech_classification.ipynb) showcases the model training, evaluation, and visualizations of the results.

### Contributors
Mohammed Anas Khan 

### License
This project is licensed under the GNU License.


