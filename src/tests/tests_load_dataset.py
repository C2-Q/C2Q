from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "boshuai1/c2q-parser-codebert"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
print("ok")

