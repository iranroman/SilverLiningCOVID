# run the sentiment analysis on Ripolles data using model
# distilbert-base-uncased-finetuned-sst-2-english
# 91% accuracy on the SST (Stanford Sentiment Treebank dataset)
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/tree/main

# load model 
import torch
import numpy as np 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("MilaNLProc/feel-it-italian-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("MilaNLProc/feel-it-italian-sentiment")

# import data
import csv
csv_file = open('silverlinings.csv','r')
sentences = []
ratings = []
language = []
location = []

# Read off and discard first line, to skip headers
csv_file.readline()

# Split columns while reading
for a, b, c, d in csv.reader(csv_file, delimiter=','):
    # Append each variable to a separate list
    sentences.append(a)
    ratings.append(b)
    language.append(c)
    location.append(d)

# remove empty entries
empty_idx = [i for i,s in enumerate(sentences) if s=='']
import numpy as np
sentences = np.delete(sentences, empty_idx).tolist()
ratings = np.delete(ratings, empty_idx).tolist()
language = np.delete(language, empty_idx).tolist()
location = np.delete(location, empty_idx).tolist()

# US english only
eng_idx = [i for i,(g,l) in enumerate(zip(language, location)) if g=='2' and l=='2']
sentences = [sentences[i] for i in eng_idx]
ratings = [ratings[i] for i in eng_idx]

# repeat the analysis to assess model consistency across 
# random samplings of the test data (Ripolles data)
ntries = 5
js = []
for n in range(ntries):
    # balance positive and negative
    pos_idx = np.argwhere(np.array(ratings).astype(np.float)==1)
    neg_idx = np.argwhere(np.array(ratings).astype(np.float)==-1)
    N = 36 # number of random samples
    pos_idx = np.random.choice(np.squeeze(pos_idx), size=N, replace=False)
    neg_idx = np.random.choice(np.squeeze(neg_idx), size=N, replace=False)
    
    j = 0.0
    for i in np.concatenate((pos_idx, neg_idx)):
        print(sentences[i])
        inputs = tokenizer(sentences[i], return_tensors='pt')
        # Call the model and get the logits
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(**inputs, labels=labels)
        loss, logits = outputs[:2]
        logits = logits.squeeze(0)

        # Extract probabilities
        proba = torch.nn.functional.softmax(logits, dim=0)

        # Unpack the tensor to obtain negative and positive probabilities
        negative, positive = proba
        print(f"Probabilities: Negative {np.round(negative.item(),4)} - Positive {np.round(positive.item(),4)}")
        print(ratings[i])

        if negative > positive and ratings[i] == '-1': 
            j += 1.0
        elif positive > negative and ratings[i] == '1': 
            j += 1.0
    
    js.append(j/(2*N))

print('The mean test accuracy is: ', np.mean(js))
print('------------- and the std: ', np.std(js))
