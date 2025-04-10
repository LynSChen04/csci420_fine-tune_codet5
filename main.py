from transformers import (
    RobertaTokenizer, T5ForConditionalGeneration,
    Trainer, TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import pickle
import pandas as pd
import data_processing
from pathlib import Path
import os 

if __name__ == "__main__":
    folder = Path("ProvidedData")
    file1 = "masked_train.csv"
    file2 = "uniqueTokens.pkl"
    #check if both files exist, if not create
    if not Path(file1).exists() or not Path(file2).exists():
        print("files do not exist, creating")
        data_processing.dataCleaning("Salesforce/codet5-base", folder/"ft_train.csv")
        
    os.makedirs("codet5-finetuned", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("getting tokenizer and model")
    #Tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
    print("reading pickle file of unique tokens")
    with open("uniqueTokens.pkl", "rb") as f:
        tokens = pickle.load(f)
    print("adding any unknown tokens")
    new_tokens = [t for t in tokens if t not in tokenizer.get_vocab()]
    tokenizer.add_tokens(new_tokens)
    tokenizer.add_tokens(["<IF-STMT>"]) #Imagine we need an extra token. This line adds the extra token to the vocabulary
    model.resize_token_embeddings(len(tokenizer))

    #cleaned data
    training = pd.read_csv("masked_train.csv")
    
    input = training["tokenized"].tolist()
    answer = training["target_block"].tolist()
    
    # Load given CSVs
    train_df = pd.read_csv("ProvidedData/ft_train.csv")
    valid_df = pd.read_csv("ProvidedData/ft_valid.csv")
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df[["cleaned_method", "target_block"]])
    valid_dataset = Dataset.from_pandas(valid_df[["cleaned_method", "target_block"]])
    
    train_dataset = train_dataset.map(data_processing.preprocess_function, batched=True)
    valid_dataset = valid_dataset.map(data_processing.preprocess_function, batched=True)

    training_args = TrainingArguments(
    output_dir="./codet5-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    logging_steps=100,
    push_to_hub=False,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )


    # --- Fine-tune automatically ---
    trainer.train()
    trainer.save_model("./codet5-finetuned/final-model")
    print("finished")
