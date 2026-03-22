import csv
import re
import random   

data = {}

with open("IMDB Dataset.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        review = row[0].strip()
        sentiment = row[1].strip().lower()
        label = True if sentiment == "positive" else False

        data[review] = label   

# shuffle by converting to list of items
items = list(data.items())
random.shuffle(items)

split = int(0.8 * len(items))
train_items = items[:split]
test_items = items[split:]

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text.strip()

# rebuild dicts
train_data = {}
for review, label in train_items:
    cleaned = clean_text(review)
    train_data[cleaned] = label   

test_data = {}
for review, label in test_items:
    cleaned = clean_text(review)
    test_data[cleaned] = label    

# write out as dicts
with open("data.py", "w", encoding="utf-8") as f:
    f.write("train_data = {\n")
    for review, label in train_data.items():
        review = review.replace('"', '\\"')
        f.write(f'  "{review}": {label},\n')
    f.write("}\n\n")

    f.write("test_data = {\n")
    for review, label in test_data.items():
        review = review.replace('"', '\\"')
        f.write(f'  "{review}": {label},\n')
    f.write("}\n")