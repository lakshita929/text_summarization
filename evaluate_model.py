# evaluate_model.py

import argparse
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from dataset_preprocessing import load_and_preprocess_dataset

def evaluate_model(model_dir, tokenizer_name):
    # Load datasets and tokenizer
    _, _, tokenizer = load_and_preprocess_dataset(50000, 5000, tokenizer_name)

    # Load the trained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    # Load the test dataset
    from datasets import load_dataset
    dataset = load_dataset("cnn_dailymail", name="3.0.0")
    test_dataset = dataset["test"]

    # Preprocess the test dataset
    def preprocess_function(examples):
        inputs = [doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length', return_tensors="pt")
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(text_target=examples["highlights"], max_length=150, truncation=True, padding='max_length', return_tensors="pt")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights"])
    tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Evaluate the model
    trainer = Seq2SeqTrainer(model=model)
    results = trainer.evaluate(tokenized_test_dataset)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument("--model_dir", type=str, default="/content/results", help="Directory containing the trained model")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/bart-base", help="Tokenizer to use")

    args = parser.parse_args()

    evaluate_model(args.model_dir, args.tokenizer_name)
