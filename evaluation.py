from transformers import T5ForConditionalGeneration
import pandas as pd
import torch
import evaluate
from transformers import AutoTokenizer
from codebleu import calc_codebleu

def evalModel(model_path, csv_path, output):
    # Load the saved model
    model_path = "codet5-finetuned/final-model"
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # Read the "tokenized" column from masked_test.csv

    csv_path = "./masked_test.csv"
    df = pd.read_csv(csv_path)
    input_tests = df["tokenized"].tolist()

    # Run the test on the model

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Generate predictions for each input
    outputs = []
    for input_tokens in input_tests:  # Assuming input_tests is now a list of tokenized inputs
        # Convert the list of tokens to input IDs
        # Ensure input_tokens is tokenized properly
        input_tokens = input_tokens.strip("[]").replace("'", "").split(", ")
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
    # for i, output in enumerate(outputs):
    #     print(f"Input {i + 1}: {input_tests[i]}")
    #     print(f"Output {i + 1}: {output}")
    #     print()


    # Load the BLEU metric
    bleu = evaluate.load("bleu")

    # Replace tokens not in the tokenizer's vocabulary with "<unk>"
    def replace_unknown_tokens(tokenizer, tokens):
        return [
            token if token in tokenizer.get_vocab() else "<unk>"
            for token in tokens
        ]

    # Prepare references and predictions
    raw_references = [tokenizer(code,return_tensors="pt",padding=True) for code in df["target_block"]]
    raw_predictions = [tokenizer(code,return_tensors="pt",padding=True) for code in outputs]
    references = [
        [tokenizer.batch_decode(ref["input_ids"], skip_special_tokens=True)[0]]
        for ref in raw_references
    ]
    predictions = [
        tokenizer.batch_decode(pred["input_ids"], skip_special_tokens=True)[0]
        for pred in raw_predictions
    ]
    codebleu_references = [
        tokenizer.batch_decode(ref["input_ids"], skip_special_tokens=True)[0]
        for ref in raw_references
    ]

    # print(len(codebleu_references[0]))
    # print(len(predictions[0]))

    # print("Type of predictions:", type(predictions))
    # print("Type of references:", type(codebleu_references))

    # print("Length of predictions:", len(predictions))
    # print("Length of references:", len(codebleu_references))

    def clean_codebleu_inputs(predictions, references):
        # Ensure strings, strip whitespace
        cleaned_preds = []
        cleaned_refs = []
        for p, r in zip(predictions, references):
            if isinstance(p, str) and isinstance(r, str):
                cleaned_preds.append(p.strip())
                cleaned_refs.append(r.strip())
            else:
                print(f"Skipping bad input pair: {type(p)} / {type(r)}")
        return cleaned_preds, cleaned_refs

    preds, refs = clean_codebleu_inputs(predictions, codebleu_references)

    # Calculate CodeBLEU for each reference and prediction pair individually
    codebleu_scores = []
    for ref, pred in zip(refs, preds):
        score = calc_codebleu([ref], [pred], lang="python")
        codebleu_scores.append(score)

    # Add CodeBLEU scores to the DataFrame
    df["codebleu_score"] = codebleu_scores

    # Print a message indicating completion
    print("CodeBLEU scores calculated")


    # Compute BLEU score
    bleu_score = bleu.compute(predictions=predictions, references=references)
    # Calculate BLEU score for each reference and prediction pair individually
    bleu_scores = []
    for ref, pred in zip(references, predictions):
        individual_bleu = bleu.compute(predictions=[pred], references=[[ref[0]]])
        bleu_scores.append(individual_bleu['bleu'])

    # Add BLEU scores to the DataFrame
    df["bleu_score"] = bleu_scores

    # Print a message indicating completion
    print("BLEU scores calculated")


    # Compute exact match score and save True/False for each prediction
    def exact_match(predictions, references):
        return [pred == ref for pred, ref in zip(predictions, references)]

    # Add exact match results to the DataFrame
    df["exact_match"] = exact_match(outputs, df["target_block"].tolist())

    # Compute overall exact match score
    exact_match_score = df["exact_match"].mean()
    print(f"Exact Match Score: {exact_match_score}")

    # Prepare the "final_results" DataFrame
    final_results = pd.DataFrame()

    # Copy "cleaned_method" and replace "target_block" with "<mask>"
    final_results["input_function"] = df["cleaned_method"].apply(
        lambda method: method.replace(df["target_block"].iloc[0], "<mask>")
    )

    final_results["expected_if"] = df["target_block"]
    final_results["predicted_if"] = outputs
    final_results["bleu_score"] = [score * 100 for score in bleu_scores]
    final_results["codebleu_score"] = [score["codebleu"] * 100 for score in codebleu_scores]
    final_results["exact_match"] = df["exact_match"]

    # Save the "final_results" DataFrame to a CSV file
    final_results.to_csv(output, index=False)
    print("Finished")

if __name__ == "__main__":
    evalModel("codet5-finetuned/final-model", "./masked_test.csv", "final_results.csv")