# Spam Detection with Advanced NLP and TensorFlow
The model architecture incorporates Bidirectional LSTM layers, GlobalMaxPooling1D, dropout for regularization, learning rate scheduling, and early stopping.

## Features
### Data Processing: 
Utilizes Pandas for loading and preprocessing spam dataset, including tokenization and padding of text sequences.
### Model Architecture: 
Employs a sophisticated neural network architecture with word embeddings, Bidirectional LSTM layers, GlobalMaxPooling1D, dropout, and dense layers.
### Callbacks: 
Implements learning rate scheduling and early stopping using TensorFlow callbacks for efficient training and prevention of overfitting.
### Evaluation Metrics: 
Computes accuracy, loss, confusion matrix, and classification report for comprehensive model evaluation.
### Learning Rate Scheduling: 
Adapts the learning rate during training to optimize model convergence.
### Early Stopping: 
Halts training when there is no improvement in validation loss to prevent overfitting.

## Dependencies
- NumPy
- Pandas
- TensorFlow
- Scikit-learn

## Usage
1. Clone the repository.
2. Install the required dependencies using pip install -r requirements.txt.
3. Run the Jupyter Notebook or Python script for training and evaluating the spam detection model.
