import re
import sys
import os

#!{sys.executable} -m spacy download en
import spacy
import scipy
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Fixing the different patterns
email_pat = r"([\w.+-]+@[a-z\d-]+\.[a-z\d.-]+)"
punct_pat = r"[,|.|_|@|\|?|\\|$&*|%|\r|\n|.:|\s+|/|//|\\|/|\||-|<|>|;|(|)|=|+|#|-|\"|[-\]]|{|}]"
num_pat = r"(?<!)(\d+(?:\.\d+)?)"

# Define a function to remove email_patterns, punctuations and numbers from the text
def preText(text):
    # Make the text unicase (lower) 
    text = str(text).lower()
    # Remove email adresses
    text = re.sub(email_pat, ' ', text, flags=re.IGNORECASE)
    # Remove all numbers
    text = re.sub(r'\d+',' ',text)# remove numbers
    text = re.sub(num_pat, ' ', text)
    # Replace all punctuations with blank space
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(punct_pat, " ", text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    # remove HTML tags
    text = re.sub('<.*?>', '', text)   
    # Replace multiple spaces from prev step to single
    text = re.sub(r' {2,}', " ", text, flags=re.MULTILINE)
    text = text.replace('`',"'")
    return text.strip()

# Initialize spacy 'en' medium model, keeping only tagger component needed for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Define a function to lemmatize the descriptions
def lemmatizer(sentence):
    # Parse the sentence using the loaded 'en' model object `nlp`
    doc = nlp(sentence)
    return " ".join([token.lemma_ for token in doc if token.lemma_ !='-PRON-'])

def clusters(X,iters):
    sse=[]
    models=[]
    scores=[]
    for i in iters:
        kmeans=KMeans(n_clusters=i)
        clusters=kmeans.fit(X)
        sse.append(kmeans.inertia_)
        scores.append(silhouette_score(X, kmeans.labels_))
    return sse, scores

#Setting up paths to load Word2Vec Data
data='DATA'
processed_data_folder='PROCESSED_DATA'
DATA_PATH=os.path.join(data)
PROCESSED_DATA_PATH=os.path.join(DATA_PATH,processed_data_folder)

word2vec_folder='word2vec'
WORD_2_VEC_PATH=os.path.join(PROCESSED_DATA_PATH,word2vec_folder)

from gensim.models import KeyedVectors
word_vecs_reload = KeyedVectors.load(os.path.join(WORD_2_VEC_PATH,'word2vec.wordvectors'), mmap='r')
with open(os.path.join(WORD_2_VEC_PATH,'dict_wordprob'),'r')as f:
    dict_word_prob=f.read()


def get_average_vector_sent(sent,dict_sent={},a=0.0001,word_vecs_reload=word_vecs_reload,dict_word_prob=dict_word_prob):
    av_vect=[]
    sum_word_count=0
    for word in sent.split():
        if word not in word_vecs_reload:
            print('word not in word_vecs_reload.keys')
        if word not in dict_word_prob.keys():
            print('word not in dict_word_prob.keys')
        vec=word_vecs_reload[word]*(a/(a+dict_word_prob[word]))
        av_vect.append(vec)
    mean=np.mean(av_vect, axis=0)
    dict_sent[sent]=mean
    return mean

def get_df_vectors(sents):
    dict_sent={}
    means=[]
    for sent in sents:
        mean=get_average_vector_sent(sent,dict_sent)
        means.append(mean)
    return np.array(means).squeeze()

def get_final_vectors(sent):
    pre_array=get_df_vectors(sent).T
    svd_u=scipy.linalg.svd(pre_array)[0]
    u2=svd_u.dot(svd_u.T).dot(pre_array)
    return pre_array-u2