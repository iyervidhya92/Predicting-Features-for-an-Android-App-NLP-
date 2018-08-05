# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 19:16:33 2017
@author: HakunaMatata
"""
import collections
import csv
from nltk.tokenize import wordpunct_tokenize
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity 
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import WordNetLemmatizer
import string
from nltk.stem.porter import PorterStemmer
from gensim.models import ldamodel
lemmatiser = WordNetLemmatizer() 
p_stemmer = PorterStemmer()
tfidf_vectorizer = TfidfVectorizer()
tokenizer = RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','``', 'the', 'and', 'to', 'of', 'a', 'I', 'in',
            'was', 'he', 'that', 'it', 'his', 'her', 'you', 'as',
            'had', 'with', 'for', 'she', 'not', 'at', 'but', 'be',
            'my', 'on', 'have', 'him', 'is', 'said', 'me', 'which',
            'by', 'so', 'this', 'all', 'from', 'they', 'no', 'were',
            'if', 'would', 'or', 'when', 'what', 'there', 'been',
            'one', 'could', 'very', 'an', 'who'])

#download reviews from csv_file
def getData(csv_file):
    data=[]
    reader = csv.reader(csv_file, delimiter=',')
    next(reader)
    for row in reader:
        if not '' in row: 
           data.append(row[2])         
    return data

#download version information from textfile
def getVersion(version_file):
    data=[]
    data = [line.split(',') for line in (version_file)]
    return data

#data-preprocessing
def my_tokenizer(reviews):
    temp=[]
    for line in reviews:
      #line = line.decode('utf-8')
      tokens = wordpunct_tokenize(line)
      tokens=filter(lambda x: x not in string.punctuation, tokens)
      tokens=filter(lambda x: x not in stop_words, tokens)
      for w in tokens:
        if w.lower not in stop_words:
            lemmatiser.lemmatize(w)
            temp.append(w)  
    return([str(temp)])

#data-preprocessing
def my_tokenizer2(data_set):   
    texts=[] 
    for i in data_set:
        i=i.decode('utf-8')
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [t for t in tokens if not t in stop_words]
        # stem tokens
        stemmed_tokens = [lemmatiser.lemmatize(t) for t in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens) 
    return(texts)

#to get the tfidf matrix and perform kmeans
def tfidf_func(reviews):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer,
                                        stop_words=stopwords.words('english'),
                                        lowercase=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    dist = 1-cosine_similarity(tfidf_matrix)

    l = np.mean(range(1,len(reviews)))
    l=int(l)
    K=range(1,20)
    meanDistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(tfidf_matrix)
        SSE = sum(np.min(cdist(dist, kmeans.cluster_centers_,'euclidean'),axis=1))/tfidf_matrix.shape[0]
        meanDistortions.append(SSE) #value of SSE
    clusters = collections.defaultdict(list)
    labels = kmeans.predict(tfidf_matrix)
    for i, label in enumerate(kmeans.labels_):
         clusters[label].append(i)
    return (dict(clusters),meanDistortions,labels)

#to print out the clusters
def print_clusters(clusters, reviews, no_of_clusters):
    n = 0
    print '========\n'
    print('Group'); print(n)
    print '========\n'
    while n < no_of_clusters:
          for q,sentence in enumerate(clusters[n]):
              print reviews[sentence]
              if n < no_of_clusters:
                 n=n+1
                 print '========\n'
                 print('Group'); print(n)
                 print '========\n'


#main_function
input_path = "C:\Python27"
filename = "CNN5.5.4 - Copy.csv" 
reviews=[] 

with open(input_path+"/"+ filename) as csv_file:
     reviews = getData(csv_file)
clusters,sse,label = tfidf_func(reviews)
for i in range(1,20):
    no_of_clusters = sse.index(min(sse))
if no_of_clusters==0:
   no_of_clusters=1
print_clusters(clusters,reviews,no_of_clusters)
temp=collections.defaultdict(list)

    
filename = "version.txt"
version = []


with open(input_path+"/"+ filename) as version_file:
    version = getVersion(version_file)
  

print '\n========'
print('Version')
print '========\n'


def func(corpus,dictionary,words):
  model = ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary)
  temp = model.id2word.doc2bow(text)
  lda_model = model[temp]
  return lda_model
 
model_list=[]
for i in version:
    #i=[t.decode('utf-8') for t in i]
    words = my_tokenizer2(i)
    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(text) for text in words]
    version_model = func(corpus,dictionary,words)
    print i
    model_list.append(version_model)

print '\n========'
print('Cluster Topics')
print '========\n'   

n=0 
while n <= no_of_clusters:
     for sentence in clusters[n]:
         temp[n].append(reviews[sentence]) 
     n+=1

model_list2=[]
model_list3=[]


for i,clist in temp.items():
    #clist=[i.decode('utf-8') for i in clist]
    words = my_tokenizer2(clist)
    dictionary = corpora.Dictionary(words)
    corpus = [dictionary.doc2bow(text) for text in words]
    model = ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary)
    cluster_model = func(corpus,dictionary,words)
    print clist
    print model.show_topics(cluster_model)
    model_list2.append(cluster_model)
    model_list3.append(model.show_topics(cluster_model))
   
    
    
    
print '\n========'
print('Comparison between Cluster topics and Version topics')
print '========\n' 

distances = []

def print_cluster_version(i,j,mylist):
    print '\n========'
    print 'version', version[i]
    print 'cluster\t', mylist[j]

for i in model_list2:
    for j in model_list:        
        if hellinger(i,j) > 0.5:
           print "\ntopic comparison distance between version and reviews"
           print hellinger(i,j)
           print_cluster_version(model_list.index(j), model_list2.index(i),model_list3)
           distances.append(hellinger(i,j))
           
        else:
            continue
        

        
        



   
                    
            
                

