from transformers import RobertaTokenizer, T5ForConditionalGeneration
import pandas as pd
import evaluate

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')

#cleaned data
training = pd.read_csv("masked_train.csv")

input = training["tokenized"].tolist()
answer = training["target_block"].tolist()

#template code TODO remove
text = "def greet(user): print(f'hello <extra_id_0>!')"
input_ids = tokenizer(text, return_tensors="pt").input_ids
# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=10)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints "user: {user.name}"


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

