# train_model.py

import argparse
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from dataset_preprocessing import load_and_preprocess_dataset

def train_model(output_dir, logging_dir, train_size, val_size, tokenizer_name):
    # Load datasets and tokenizer
    tokenized_train_dataset, tokenized_val_dataset, tokenizer = load_and_preprocess_dataset(train_size, val_size, tokenizer_name)

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

    # Define data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, 
        evaluation_strategy="steps", 
        fp16=True,
        gradient_accumulation_steps=8, 
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4, 
        weight_decay=0.01,
        logging_dir=logging_dir, 
        logging_steps=10, 
        save_total_limit=3, 
        num_train_epochs=3,
        load_best_model_at_end=True, 
        save_steps=500
    )

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save the trained model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--output_dir", type=str, default="/content/results", help="Directory to save the model")
    parser.add_argument("--logging_dir", type=str, default="/content/logs", help="Directory to save the logs")
    parser.add_argument("--train_size", type=int, default=50000, help="Size of the training dataset")
    parser.add_argument("--val_size", type=int, default=5000, help="Size of the validation dataset")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/bart-base", help="Tokenizer to use")
    
    args = parser.parse_args()
    
    train_model(args.output_dir, args.logging_dir, args.train_size, args.val_size, args.tokenizer_name)
