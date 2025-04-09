import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer
def dataCleaning(modelName, csv):
    # Load the pretrained CodeT5 model and tokenizer
    model_name = modelName
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    #Read files
    train_data = pd.read_csv(csv)
    print("train_data", train_data.head())


    #Data work
    train_data['tokens'] = train_data['cleaned_method'].str.rstrip()
    print("train_data['tokens']", train_data['tokens'].head())

    train_data['tokens'] = train_data.apply(
        lambda row: row['tokens'].strip().replace(row['target_block'].strip()[:-2], "<mask>") if pd.notnull(row['target_block']) else row['tokens'], 
        axis=1
    )
    print("train_data['tokens']", train_data['tokens'].head())

    #Tokenization with a preset tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_data['tokenized'] = train_data['tokens'].apply(lambda x: tokenizer.tokenize(x))

    print("train_data['tokenized']", train_data['tokenized'].head())

    # Remove non-ASCII characters from each string in the 'tokenized' column
    def remove_non_ascii(tokens):
        return [''.join(char for char in token if ord(char) < 128) for token in tokens]

    train_data['tokenized'] = train_data['tokenized'].apply(remove_non_ascii)

    print("train_data['tokenized']", train_data['tokenized'].head())

    # Combine "<", "mask", ">" into "<mask>, make sure to remove the "\n" character from the string"
    def combine_mask_tokens(tokens):
        combined_tokens = []
        i = 0
        while i < len(tokens):
            if i + 2 < len(tokens) and tokens[i] == "<" and tokens[i + 1] == "mask" and tokens[i + 2] == ">":
                combined_tokens.append("<mask>".strip())
                i += 3
            else:
                combined_tokens.append(tokens[i].strip())
                i += 1
        return combined_tokens

    train_data['tokenized'] = train_data['tokenized'].apply(combine_mask_tokens)
    print("train_data['tokenized']", train_data['tokenized'].head())

    # Add a column to the dataframe that counts the number of items in 'tokenized'
    train_data['token_count'] = train_data['tokenized'].apply(len)

    # Write the dataframe to a CSV file excluding the 'tokens' column
    train_data.drop(columns=['tokens']).to_csv("masked_train.csv", index=False)

if __name__ == "__main__":
    dataCleaning("Salesforce/codet5-base", "ft_train.csv")