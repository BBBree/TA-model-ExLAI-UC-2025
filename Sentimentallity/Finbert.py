# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Model details
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example text for the model
comments = [
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

# Sums to calculate averages
positive_sum = 0
negative_sum = 0
neutral_sum = 0

print()

for text in comments:
    results = pipe(text)
    for score in results[0]:
        print(f"{score["label"]}: {score["score"]}")

        # Selects label of the score: Positive, Negative, or Neutral
        score_label = score["label"]
        
        # Similar to a "switch" statement
        # Here, label is determined as being either positive, negative, or neutral
        # Afterwards, the value of the score is added to the sum
        match score_label: 
            case "positive":
                positive_sum += score["score"]
            case "negative":
                negative_sum += score["score"]
            case "neutral":
                neutral_sum = score["score"]
    print()

# Averages for each category
positive_avg = positive_sum / len(comments)
negative_avg = negative_sum / len(comments)
neutral_avg = neutral_sum / len(comments)

print(f"Averages from the {len(comments)} comments:")
print(f"Positive: {positive_avg}")
print(f"Negative: {negative_avg}")
print(f"Neutral: {neutral_avg}")