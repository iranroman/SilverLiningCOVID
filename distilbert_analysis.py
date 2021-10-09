# run the sentiment analysis on Ripolles data using model
# distilbert-base-uncased-finetuned-sst-2-english
# 91% accuracy on the SST (Stanford Sentiment Treebank dataset)
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/tree/main

# load model 
from transformers import pipeline
classifier = pipeline('sentiment-analysis')

# import data
import csv
csv_file = open('silverlinings.csv','r')
sentences = []
ratings = []
language = []

# Read off and discard first line, to skip headers
csv_file.readline()

# Split columns while reading
for a, b, c, _ in csv.reader(csv_file, delimiter=','):
    # Append each variable to a separate list
    sentences.append(a)
    ratings.append(b)
    language.append(c)

# remove empty entries
empty_idx = [i for i,s in enumerate(sentences) if s=='']
import numpy as np
sentences = np.delete(sentences, empty_idx).tolist()
ratings = np.delete(ratings, empty_idx).tolist()
language = np.delete(language, empty_idx).tolist()

# english only
eng_idx = [i for i,g in enumerate(language) if g=='3']
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
    N = 100 # number of random samples
    pos_idx = np.random.choice(np.squeeze(pos_idx), size=N, replace=False)
    neg_idx = np.random.choice(np.squeeze(neg_idx), size=N, replace=False)
    
    j = 0.0
    for i in np.concatenate((pos_idx, neg_idx)):
        print(sentences[i])
        print(classifier(sentences[i])[0], ratings[i])
        if classifier(sentences[i])[0]['label'] == 'NEGATIVE' and ratings[i] == '-1': 
            j += 1.0
        elif classifier(sentences[i])[0]['label'] == 'POSITIVE' and ratings[i] == '1': 
            j += 1.0
    
    js.append(j/(2*N))

print('The mean test accuracy is: ', np.mean(js))
print('------------- and the std: ', np.std(js))
