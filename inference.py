# inference.py

import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def summarize_text(model_dir, tokenizer_name, input_text):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    # Tokenize the input text
    inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and print the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Summarization using Fine-Tuned Model")
    parser.add_argument("--model_dir", type=str, default="/content/results", help="Directory containing the fine-tuned model")
    parser.add_argument("--tokenizer_name", type=str, default="facebook/bart-base", help="Tokenizer to use")
    parser.add_argument("--input_text", type=str, required=True, help="Text to summarize")

    args = parser.parse_args()

    summary = summarize_text(args.model_dir, args.tokenizer_name, args.input_text)
    print("Summary:", summary)
