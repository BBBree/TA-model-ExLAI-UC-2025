# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Model details
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example text of the model
text = [
    "Apple's stocks are soaring today, it looks like its a good time to sell.",
    "It doesn't look good for Apple, the outlook doesn't look good.",
    "People are very much loving Apple as a company, showing a huge increase in sales."

    ]

# Pipeline, used for "piping" the text to the model for analysis
# Note: top_k=None = return_all_scores=True
pipe = pipeline(
    "sentiment-analysis", 
    model = model, 
    tokenizer = tokenizer, 
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

for i in text:
    results = pipe(i)
    for score in results[0]:
        print(f"{score["label"]}: {score["score"]}")
    print()