from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
df = pd.read_csv("data/toxic_data.csv")

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# Multilingual BERT model
model_name = "bert-base-multilingual-cased"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize text
train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding=True
)

test_encodings = tokenizer(
    list(test_texts),
    truncation=True,
    padding=True
)

# Create datasets
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": list(train_labels)
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": list(test_labels)
})

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
)

# Metrics function
def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Evaluate model
results = trainer.evaluate()

print(results)

# Save model
model.save_pretrained("toxicity_model")
tokenizer.save_pretrained("toxicity_model")

print("Training completed")