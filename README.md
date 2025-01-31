# HandiDigit-Classifier-CNN-Based-Handwritten-Digit-Recognition-System

## Overview

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits using the MNIST dataset. The model achieves high accuracy in recognising digits from 0 to 9, demonstrating the effectiveness of CNNs for image classification tasks.

## Features

- Data preprocessing and augmentation
- CNN architecture with convolutional layers, max pooling, and dropout
- Training with early stopping and learning rate reduction
- Evaluation on test set
- Visualisation of training history and model performance

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/handidigit-classifier.git
   ```

2. Install the required packages:
   ```
   !pip install -r requirements.txt
   ```

## Usage

1. Run the Jupyter notebook or Python script to train the model:
   ```
   jupyter notebook HandiDigit_Classifier.ipynb
   ```

2. The script will automatically download the MNIST dataset (https://paperswithcode.com/dataset/mnist), preprocess the data, train the model, and evaluate its performance.

## Model Architecture

The CNN model consists of:
- Two convolutional layers with ReLU activation
- Max pooling layers
- Flatten layer
- Dense layer with dropout for regularisation
- Output layer with softmax activation

## Results

The model achieves an accuracy of approximately 98.93% on the test set, demonstrating its effectiveness in recognising handwritten digits.

## Future Improvements

- Experiment with different CNN architectures
- Implement data augmentation techniques
- Try transfer learning with pre-trained models
- Develop a web interface for real-time digit recognition

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or contac me to any colaboration: brunolopessousa23@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- The MNIST dataset providers
- TensorFlow and Keras development teams
