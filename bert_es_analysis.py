# run the sentiment analysis on Ripolles data using model
# distilbert-base-uncased-finetuned-sst-2-english
# 91% accuracy on the SST (Stanford Sentiment Treebank dataset)
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/tree/main

# load model 
import torch
from scipy.special import softmax

from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")




# import data
import csv
csv_file = open('silverlinings_final.csv','r') # distilbert-base-uncased-finetuned-sst-2-english
sentences = []
ratings = []
language = []
location = []

# Read off and discard first line, to skip headers
csv_file.readline()

# Split columns while reading
for a, b, c, d, _, _, _, _, _, _ in csv.reader(csv_file, delimiter=','):
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

# Spain Spanish only
eng_idx = [i for i,(g,l) in enumerate(zip(language, location)) if g=='1' and (l=='2')]
sentences = [sentences[i] for i in eng_idx]
ratings = [ratings[i] for i in eng_idx]

# repeat the analysis to assess model consistency across 
# random samplings of the test data (Ripolles data)
ntries = 1000
js = []
for n in range(ntries):
    # balance positive and negative
    pos_idx = np.argwhere(np.array(ratings).astype(np.float)==1)
    neg_idx = np.argwhere(np.array(ratings).astype(np.float)==-1)
    N = 10 # number of random samples
    pos_idx = np.random.choice(np.squeeze(pos_idx), size=N, replace=False)
    neg_idx = np.random.choice(np.squeeze(neg_idx), size=N, replace=False)
    
    j = 0.0
    for i in np.concatenate((pos_idx, neg_idx)):
        print(sentences[i])

        inputs = tokenizer(sentences[i], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = softmax(logits).numpy()[0]
        pos_side = np.sum(probabilities[3:])
        neg_side = np.sum(probabilities[:2])
        #print(probabilities)
        #print(pos_side, neg_side)
        #print(ratings[i])
        #print('')
        #input()

        if neg_side > pos_side and ratings[i] == '-1': 
            j += 1.0
        elif pos_side > neg_side and ratings[i] == '1': 
            j += 1.0
    
    js.append(j/(2*N))

print('The mean test accuracy is: ', np.mean(js))
print('------------- and the std: ', np.std(js))


#The mean test accuracy is:  0.9334499999999999
#------------- and the std:  0.05318456072959519
