# run the sentiment analysis on Ripolles data using model
# distilbert-base-uncased-finetuned-sst-2-english
# 91% accuracy on the SST (Stanford Sentiment Treebank dataset)
# https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/tree/main

# load model 
from pysentimiento import create_analyzer
classifier = create_analyzer(task="sentiment", lang="es")

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
# remove NaNs entries
NaN_idx = [i for i,s in enumerate(ratings) if s=='NaN']
sentences = np.delete(sentences, NaN_idx).tolist()
ratings = np.delete(ratings, NaN_idx).tolist()
language = np.delete(language, NaN_idx).tolist()
location = np.delete(location, NaN_idx).tolist()

# spanish only
es_idx = [i for i,(g,l) in enumerate(zip(language, location)) if g=='1' and (l=='1' or l=='5')]
sentences = [sentences[i] for i in es_idx]
ratings = [ratings[i] for i in es_idx]

# repeat the analysis to assess model consistency across 
# random samplings of the test data (Ripolles data)
ntries = 5
js = []
for n in range(ntries):
    # balance positive and negative
    pos_idx = np.argwhere(np.array(ratings).astype(np.float)==1)
    neg_idx = np.argwhere(np.array(ratings).astype(np.float)==-1)
    N = 34 # number of random samples
    pos_idx = np.random.choice(np.squeeze(pos_idx), size=N, replace=False)
    neg_idx = np.random.choice(np.squeeze(neg_idx), size=N, replace=False)
    
    j = 0.0
    for i in np.concatenate((pos_idx, neg_idx)):
        print(sentences[i])
        model_out = classifier.predict(sentences[i])
        print(model_out.output, ratings[i])
        if model_out.output == 'NEG' and ratings[i] == '-1': 
            j += 1.0
        elif model_out.output == 'POS' and ratings[i] == '1': 
            j += 1.0
        elif model_out.output == 'NEU': 
            if model_out.probas['NEG'] > model_out.probas['POS'] and ratings[i] == '-1':
                j += 1.0
            elif model_out.probas['NEG'] < model_out.probas['POS'] and ratings[i] == '1':
                j += 1.0

    js.append(j/(2*N))

print('The mean test accuracy is: ', np.mean(js))
print('------------- and the std: ', np.std(js))
