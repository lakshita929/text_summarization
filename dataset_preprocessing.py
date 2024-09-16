# dataset_preprocessing.py
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_dataset(train_size, val_size, tokenizer_name):
    # Load the CNN/DailyMail dataset
    dataset = load_dataset("cnn_dailymail", name="3.0.0")
    
    # Select a smaller subset of the training and validation sets
    small_train_dataset = dataset["train"].select(range(train_size))
    small_val_dataset = dataset["validation"].select(range(val_size))
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenizing the input text and summaries
    def preprocess_function(examples):
        inputs = [doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length', return_tensors="pt")
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(text_target=examples["highlights"], max_length=150, truncation=True, padding='max_length', return_tensors="pt")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply preprocessing to the datasets
    tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights"])
    tokenized_val_dataset = small_val_dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights"])
    
    # Ensure the dataset columns are properly set for PyTorch models
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_train_dataset, tokenized_val_dataset, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Preprocessing for Summarization Model")
    parser.add_argument("--train_size", type=int, default=50000, help="Size of the training dataset")
    parser.add_argument("--val_size", type=int, default=5000, help="Size of the validation dataset")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/bart-base", help="Tokenizer to use")
    
    args = parser.parse_args()
    
    tokenized_train_dataset, tokenized_val_dataset, tokenizer = load_and_preprocess_dataset(
        args.train_size, args.val_size, args.tokenizer_name
    )
