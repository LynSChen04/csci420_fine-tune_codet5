# GenAI for Software Development (Fine-Tuning CodeT5 for Predicting if Statements)

- [1 Introduction](#1-introduction)
- [2 Getting Started](#2-getting-started)
  - [2.1 Preparations](#21-preparations)
  - [2.2 Install Packages](#22-install-packages)
- [3 Model Training & Evaluation](#3-model-training--evaluation)
  - [3.1 Finetuning Code-t5](#31-finetuning-code-t5)
  - [3.2 Dataset Preparation](#32-dataset-preparation)
  - [3.3 Model Evaluation](#33-model-evaluation)
  - [3.4 Finetuning Metrics](#34-finetuning-metrics)
- [4 Report](#4-report)

---

# **1. Introduction**

This project finetunes the CodeT5 model to predict python if statements. By default it will go through 5 epochs and save relevant data about the training to the logs folder.

---

# **2. Getting Started**

This project is implemented in **Python 3.12+** and is compatible with **macOS, Linux, and Windows**.

## **2.1 Preparations**

(1) Clone the repository to your workspace:

```shell
~ $ git clone https://github.com/LynSChen04/csci432_fine-tune_codet5
```

(2) Navigate into the repository:

```
~ $ cd csci432_fine-tune_codet5
~/csci432_fine-tune_codet5 $
```

(3) Set up a virtual environment and activate it:

For macOS/Linux:

```
~/csci432_fine-tune_codet5 $ python -m venv ./venv/
~/csci432_fine-tune_codet5 $ source venv/bin/activate
(venv) ~/csci432_fine-tune_codet5 $
```

For Windows:

```
~/csci432_fine-tune_codet5 $ python -m venv ./venv/
~/csci432_fine-tune_codet5 $ .\venv\Scripts\activate
```

To deactivate the virtual environment, use the command:

```
(venv) $ deactivate
```

## **2.2 Install Packages**

Install the required dependencies:

```shell
(venv) ~/csci432_fine-tune_codet5 $ pip install -r requirements.txt
```

## **3.1 Finetuning Code-t5**

The `main.py` script takes a corpus of Python methods titled `masked_train.csv`, and a pickle file titled `uniqueTokens.pkl`. We use the pickle files to expand the tokenizers vocabulary so that every token is guranteed known and recongized. We then use `masked_train.csv` as our training corpus, `\ProvidedData\ft_test.csv`, and `\ProvidedData\ft_valid.csv` as our testing and validation sets respectively. Our `main.py` file saves relevant model data including the best model, which is determined by metrics that will be discussed later, to `csci432_fine-tune_codet5\codet5-finetuned`.

## **3.2 Dataset Preparation**

As the assignment dictated, we used a pretrained tokenizer from "Salesforce/CodeT5-base" to tokenize the training and validating data. Additionally, for the test dataset we removed whitespace and masked the target if statement of the input function with a masked token, ensuring that after the entire process that the token would stay as one instead of separating. This would then serve as the proper input for our evaluation where the model would have to predict the contents of what the masked token should be.

## **3.3 Model Evaluation**

The `evaluation.py` script takes the locally saved model's directory and an already masked test dataset that has been run through the original `data_processing.py` script. It will then produce `final_results.csv`, which will contain the inputted function, the expected if statement, the model's predicted if statement, the BLEU score, the CodeBLEU score, and whether it was an exact match. Additionally, the script will also print out the overall BLEU, CodeBLEU, and exact match metrics. For the outputted file, BLEU and CodeBLEU scores are rescaled to be from 0 to 100.

## **3.4 Finetuning Metrics**

We did not modify the given parameters extensively. The parameter we used to determine the best model was the loss function on the validation set.

## 4. Report

The assignment report is available in the file **Writeup.pdf**.
The csv file with our test set results is available in the file **final_results.csv**, with required True/False column for exact match, expected if condition, predicted if condition, CodeBLEU prediction score on a scale of 0 to 100, and BLEU prediction score on a scale of 0 to 100.
