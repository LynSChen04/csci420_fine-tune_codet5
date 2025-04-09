from transformers import (
    RobertaTokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Trainer, TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import pickle
import pandas as pd
import evaluate
import data_processing
from pathlib import Path

#contains all evaluation things
def modelEvaluations():
    
    #Model evaluation:
    bleu = evaluate.load("bleu")  # This uses BLEU-4 by default (n_gram=4)
    sacrebleu = evaluate.load("sacrebleu")
    return("ping")

#helper function for exact correctness
def exact(predict, actual):
    correct = 0
    for p,a in zip(predict, actual):
        if p == a:
            correct += 1
    return float(correct)/len(predict)

if __name__ == "__main__":
    folder = Path("ProvidedData")
    file1 = "masked_train.csv"
    file2 = "uniqueTokens.pkl"
    #check if both files exist, if not create
    if not Path(file1).exists() or not Path(file2).exists():
        print("files do not exist, creating")
        data_processing.dataCleaning("Salesforce/codet5-base", folder/"ft_train.csv")

    print("getting pretrained")
    #Tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
    with open("uniqueTokens.pkl", "rb") as f:
        tokens = pickle.load(f)
    print(len(tokens))
    tokenizer.add_tokens(["<IF-STMT>"]) #Imagine we need an extra token. This line adds the extra token to the vocabulary
    model.resize_token_embeddings(len(tokenizer))
    #cleaned data
    training = pd.read_csv("masked_train.csv")
    
    print("getting from csv")
    input = training["tokenized"].tolist()
    answer = training["target_block"].tolist()
    
    '''
    # --- Config ---
    MODEL_NAME = "Salesforce/codet5-small"
    DATA_PATH = "your_file.csv"  # contains "docstring" and "code"
    OUTPUT_DIR = "./codet5-finetuned"
    MAX_LENGTH = 128
    EPOCHS = 5
    BATCH_SIZE = 8
    PATIENCE = 3


    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    # --- Fine-tune automatically ---
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final-model")

    '''
