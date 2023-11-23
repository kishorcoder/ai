import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) 

# Bad words dataset 
bad_word_list = ["offensive", "terrible","jigaboo","mound of venus","asslover","s&m","fucker","queaf","whitetrash","meatrack"]

# bad words in a comment and display them
def predict_bad_words_with_display(comment):
    tokens = tokenizer.encode_plus(comment, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    
    # Use the model to predict
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    if predicted_class == 1:  # Bad detected
        # Check for bad words in the comment
        for word in bad_word_list:
            if word in comment.lower():
                return f"Bad word '{word}' found in the comment: '{comment}'"
    else:
        return "No bad words detected"

comments_dataset = [
    "This is a normal comment without any bad words.",
    "I can't believe they used such offensive language!",
    "The product is terrible.",
]

for comment in comments_dataset:
    result = predict_bad_words_with_display(comment)
    print(result)

new_comment = input()
result = predict_bad_words_with_display(new_comment)
print(result)
