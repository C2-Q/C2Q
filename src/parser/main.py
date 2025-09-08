import numpy as np
from datasets import load_dataset, load_metric
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, \
    AutoModelForSequenceClassification
import ast

# Load my dataset
dataset = load_dataset('csv', data_files={'json': 'json.csv'})
# CodeBERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


def tokenize_function(examples):
    code_snippet = examples['code_snippet']
    return tokenizer(code_snippet, padding="max_length", truncation=True)


tokenized_datasets = dataset['json'].map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=6)
training_args = TrainingArguments(output_dir="../../others/test_trainer",
                                  eval_strategy="epoch",
                                  num_train_epochs=10)

# Split the tokenized dataset into train (80%) and test (20%) sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.2, shuffle=True)

# Assign train and test datasets
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
results = trainer.evaluate()
print(results)

import torch

print(torch.backends.mps.is_available())  # Should return True if MPS is supported.
print(torch.backends.mps.is_built())  # Should return True if PyTorch is built with MPS support.
device = torch.device("cpu")
model.to(device)

labels = ["MaxCut", "MIS", "TSP", "Clique", "KColor", "Factor","ADD", "MUL", "SUB", "Unknown"]


def predict(code: str):
    inputs = tokenizer(code, return_tensors="pt", padding="max_length", truncation=True).to(device)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)  # Get probabilities
    max_prob, prediction = torch.max(probabilities, dim=-1)
    return labels[prediction.item()]


# Run prediction on the evaluation set and calculate accuracy manually
correct = 0
for i in range(len(eval_dataset)):
    code = eval_dataset[i]['code_snippet']
    true_label = eval_dataset[i]['labels']
    predicted_label = predict(code)
    if predicted_label == labels[true_label]:
        correct += 1
    print(f"True: {labels[true_label]}, Predicted: {predicted_label}")

# Calculate and print accuracy manually
accuracy = correct / len(eval_dataset)
print(f"Manual Accuracy: {accuracy * 100:.2f}%")
