from transformers import T5ForConditionalGeneration
import pandas as pd
from transformers import T5Tokenizer
import torch
import evaluate
import subprocess

# Load the saved model
model_path = "./codet5-finetuned/final-model"
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Read the "tokenized" column from masked_test.csv

csv_path = "./masked_test.csv"
df = pd.read_csv(csv_path)
input_tests = df["tokenized"].tolist()

# Run the test on the model

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Generate predictions for each input
outputs = []
for input_tokens in input_tests:  # Assuming input_tests is now a list of tokenized inputs
    # Convert the list of tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_ids = torch.tensor([input_ids])  # Convert to a PyTorch tensor with batch dimension

    # Generate predictions
    output_ids = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    outputs.append(output_text)

    # Save the outputs to the CSV file with a new column "model_output"
    df["model_output"] = outputs
    df.to_csv(csv_path, index=False)

# Print the outputs
for i, output in enumerate(outputs):
    print(f"Input {i + 1}: {input_tests[i]}")
    print(f"Output {i + 1}: {output}")
    print()


# Evaluation scores

# Load the BLEU metric
bleu = evaluate.load("bleu")

references = [[ref.split()] for ref in df["reference"].tolist()]
predictions = [pred.split() for pred in outputs]

# Compute BLEU score
bleu_score = bleu.compute(predictions=predictions, references=references)

# Print BLEU score
print(f"BLEU Score: {bleu_score['bleu']}")

#Load the SacreBLEU metric
sacrebleu = evaluate.load("sacrebleu")

# Compute SacreBLEU score
sacrebleu_score = sacrebleu.compute(predictions=predictions, references=references)

# Print SacreBLEU score
print(f"SacreBLEU Score: {sacrebleu_score['score']}")

# Compute exact match score
def exact_match(predictions, references):
    correct = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            correct += 1
    return correct / len(predictions)

# Print exact match score
exact_match_score = exact_match(outputs, df["target_block"].tolist())

# Save references and predictions to files
df["reference"].to_csv("refs.txt", index=False, header=False)
with open("preds.txt", "w") as f:
    for output in outputs:
        f.write(output + "\n")

# Run the CodeBLEU evaluator
command = [
    "python",
    "CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU/calc_code_bleu.py",
    "--refs", "refs.txt",
    "--hyp", "preds.txt",
    "--lang", "java",
    "--params", "0.25,0.25,0.25,0.25"
]
result = subprocess.run(command, capture_output=True, text=True)

# Print the CodeBLEU score
print(result.stdout)