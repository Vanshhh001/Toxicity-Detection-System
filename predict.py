from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

# Load model
model_path = "toxicity_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Prediction function
def predict_toxicity(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    outputs = model(**inputs)

    # Convert logits into probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    # Get prediction index
    prediction = torch.argmax(probabilities).item()

    # Confidence score
    confidence = probabilities[0][prediction].item() * 100

    if prediction == 1:
        label = "Toxic"
    else:
        label = "Non-Toxic"

    return label, confidence

# User input
text = input("Enter comment: ")

label, confidence = predict_toxicity(text)

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}%")