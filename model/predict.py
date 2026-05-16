from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

texts = [
    "you are useless",
    "great video bro",
    "go die"
]

for text in texts:
    result = classifier(text)[0]

    print("Text :", text)
    print("Label:", result["label"])
    print("Score:", round(result["score"], 4))
    print("-" * 30)