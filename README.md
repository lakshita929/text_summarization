# Text Summarization using BART
This project provides a workflow for training, evaluating, and running inference on a text summarization model using Hugging Face's transformers library and datasets. The model is based on Facebook's BART (Bidirectional and Auto-Regressive Transformers), specifically fine-tuned on the CNN/DailyMail dataset for the task of text summarization.

Project Overview
This project demonstrates how to:

Load and preprocess the CNN/DailyMail dataset for text summarization.
Fine-tune a pre-trained BART model using the Hugging Face transformers library.
Evaluate the model's performance on a test set.
Run inference to summarize new input text using the fine-tuned model.
We use argument parsers to make the paths and settings configurable, ensuring that anyone using the project can easily adjust parameters and paths.

Requirements
Ensure that you have the following dependencies installed:

Python 3.7+
transformers==4.30.2
datasets==2.11.0
torch==2.0.1
You can install these dependencies using the requirements.txt file provided.

Directory Structure
The project consists of the following scripts:

dataset_preprocessing.py: Handles dataset loading and preprocessing.
train_model.py: Used to fine-tune the BART model.
evaluate_model.py: Evaluates the fine-tuned model on the test set.
inference.py: Runs inference to generate summaries for new input text.
requirements.txt: Contains the necessary dependencies for the project.

Installation
Clone the repository and install the dependencies:
git clone the repo
cd the repo folder
pip install -r requirements.txt

How to Run
1. Train the Model
You can train the model using the train_model.py script. Adjust the paths for the output directory and logging directory as needed. The training set and validation set sizes can also be customized through command-line arguments.
python train_model.py --output_dir /path/to/save_model --logging_dir /path/to/logs --train_size 50000 --val_size 5000
--output_dir: Directory where the fine-tuned model will be saved.
--logging_dir: Directory for saving training logs.
--train_size: Size of the training dataset.
--val_size: Size of the validation dataset.
--tokenizer_name: Tokenizer to use (default: facebook/bart-base).
The model will be saved in the specified output_dir after training.

2. Evaluate the Model
Once the model has been trained, you can evaluate its performance on the test set using the evaluate_model.py script:
python evaluate_model.py --model_dir /path/to/saved_model --tokenizer_name facebook/bart-base
--model_dir: Directory containing the fine-tuned model.
--tokenizer_name: Tokenizer to use (default: facebook/bart-base).
The script will print the evaluation results on the test dataset.

3. Inference (Text Summarization)
To generate summaries for new text input, use the inference.py script. Pass the fine-tuned model directory and the input text you wish to summarize:
python inference.py --model_dir /path/to/saved_model --input_text "Your input text here"
--model_dir: Directory containing the fine-tuned model.
--tokenizer_name: Tokenizer to use (default: facebook/bart-base).
--input_text: The text you want to summarize.
The generated summary will be printed to the console.
