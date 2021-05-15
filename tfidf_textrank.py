#for matrix generation and creation purposes
import numpy as np
#import pandas as pd
#for data cleaning
import re
#for data processing and data cleaning
import nltk
from nltk.tokenize import sent_tokenize , word_tokenize
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
from nltk import WordNetLemmatizer
lemma = WordNetLemmatizer()
#for calculation purposes
import math
#for score generation and summary purposes 
#from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
#import matplotlib.pyplot as plt
#import seaborn as sns


def read_file(filepath):
    text_file = open(filepath)
    text = text_file.read()
    #print(text)
    return text

# text = read_file("msft.txt")
# text

def remove_shortword(sentences):
  
    words=sentences.split()
    for i in range(len(words)):
        words[i]= re.sub(r"n\'t", " not", words[i])
        words[i] = re.sub(r"\'re", " are", words[i])
        words[i] = re.sub(r"\'s", " is", words[i])
        words[i] = re.sub(r"\'d", " would", words[i])
        words[i] = re.sub(r"\'ll", " will", words[i])
        words[i] = re.sub(r"\'t", " not", words[i])
        words[i] = re.sub(r"\'ve", " have", words[i])
        words[i] = re.sub(r"\'m", " am", words[i])
    words=" ".join(words)
    return words

def sentence_token(text):
    sentences = sent_tokenize(text)
    sent_list=[]
    for sent in sentences:
        sent = sent.strip()
        sent = sent.lower()
        sent = remove_shortword(sent)
        sent_list.append(sent)
    return sent_list

def remove_punctuations(token_sent):
    text_no_ex_punc = []                                  #                            #
    for sent in token_sent:                               # removing extra punctuation #
        text_no_ex_punc.append(sent.replace('-', ' ')) #                            # 
    
    text_no_punc = []                                     #                            #
    for sent in text_no_ex_punc:                               # removing all punctuation   #
        text_no_punc.append(re.sub(r'[^\w\s]', '', sent)) #                            # 
        
    return text_no_punc

def remove_stopwords(sentences):
    cleaned_sent=[]
    for sents in sentences:
        words = word_tokenize(sents)
        sent = ""
        for word in words:
            word = word.lower() 
            if word in stopWords:
                continue
                
            if word not in stopWords:
                sent += word+" "
        cleaned_sent.append(sent)
                
    return cleaned_sent

def remove_stemming(sentences):
    
    sent_list=[]
    for sent in sentences:
        words=word_tokenize(sent)
        #words=[stemmer.stem(word) for word in words]
        words=[lemma.lemmatize(word) for word in words]
        words=" ".join(words)
        sent_list.append(words)
    return sent_list



def data_cleaning(text):
    sentences = re.sub("\s+"," ", text) # removing space #
    sentences = sent_tokenize(sentences) # tokenizing sentence #
        
    cleaned_sent = remove_punctuations(sentences)
    cleaned_sent = remove_stopwords(cleaned_sent)
    cleaned_sent = remove_stemming(cleaned_sent)
        
    return cleaned_sent



def unique_word(sentences):
    unique=[]
    for sents in sentences:
        for word in word_tokenize(sents):       
            if word not in unique:
                unique.append(word)
                
    return unique


# # Funtion for counting words per sentences

def word_count(sentence):
    count=0
    words = word_tokenize(str(sentence))
    for word in words:
            if word in stopWords:
                continue
            else:
                 count = count+1
    
    return count


# # Function for generating doc, i.e sentences details

def get_doc(cleaned_sentences):
    doc_info = []
    ind = 0
    
    for sentences in cleaned_sentences:
        ind = ind+1 #starting index from 1
        count = word_count(sentences)
        #doc_id = sentence_id
        #no_of_terms = total no of words in sentences after cleaning
        details = {"doc_id" : ind, "doc_length" : count} 
        doc_info.append(details)
    
    return doc_info



def create_freq_matrix(sentences):
    i=0
    freq_dict_list = []
    stopWords = set(stopwords.words("english"))
    for sent in sentences:
        i=i+1
        freq_dict = {}
        words = word_tokenize(sent)
        for word in words:
            if word in stopWords:
                continue
                
            if word in freq_dict:
                freq_dict[word] +=1
            else: 
                freq_dict[word] = 1
            temp = {"doc_id":i, "freq_dict":freq_dict}
        freq_dict_list.append(temp)
    
    return freq_dict_list



def tf_matrix(doc_info, unique_words, freq_dict_list):
    TF_scores = np.zeros(shape=(len(doc_info), len(unique_words)))
    for temp_dict in freq_dict_list:
        ind = temp_dict['doc_id']
        for term in temp_dict['freq_dict']:
            temp = temp_dict['freq_dict'][term] / doc_info[ind-1]['doc_length']
            TF_scores[ind-1][unique_words.index(term)] = temp
    return TF_scores



def idf_matrix(doc_info, unique_words, freq_dict_list):
    IDF_scores = np.zeros(shape=(len(doc_info), len(unique_words)))
    for temp_dict in freq_dict_list:
        ind = temp_dict['doc_id']
        for term in temp_dict['freq_dict'].keys():
            count = sum([term in tempDict['freq_dict'] for tempDict in freq_dict_list])
            temp = math.log(len(doc_info)/count)
            IDF_scores[ind-1][unique_words.index(term)] = temp
    return IDF_scores



def TFIDF_scores(doc_info,unique_words,TF_scores,IDF_scores):
    TFIDF_scores = np.zeros(shape=(len(doc_info), len(unique_words)))
    TFIDF_scores = np.multiply(TF_scores,IDF_scores)
    
    return TFIDF_scores




def cosine_similarity(sent_1,sent_2):
    s1_dot_s2=np.sum(np.multiply(sent_1,sent_2))
    magnitude_of_s1=math.sqrt(np.sum(np.multiply(sent_1,sent_1)))
    magnitude_of_s2=math.sqrt(np.sum(np.multiply(sent_2,sent_2)))
    return s1_dot_s2/(magnitude_of_s1*magnitude_of_s2)

def cosine_matrix(token_sent,tfidf_values):
    cosine_similarity_matrix=np.zeros(shape=(len(token_sent),len(token_sent)))
    for i in range(len(tfidf_values)):
        for j in range(len(tfidf_values)):
            cosine_similarity_matrix[i][j]=cosine_similarity(tfidf_values[i],tfidf_values[j])
            #print(cosine_similarity_matrix[i][j])
  
    #normalize the matrix
    for idx in range(len(cosine_similarity_matrix)):
        cosine_similarity_matrix[idx] /= cosine_similarity_matrix[idx].sum()
    
    return cosine_similarity_matrix



def textrank(cosine_similarity_matrix):
    nx_graph = nx.from_numpy_array(cosine_similarity_matrix)
    scores = nx.pagerank(nx_graph, max_iter=600)
    
    return scores







import operator
def sent_rank(token_sent, scores,n):
    sorted_d = dict( sorted(scores.items(), key=operator.itemgetter(1),reverse=True))   
    sorted_d
    dt = sorted_d
    ans=""
    #for c in range(0,n):
    for i,(j,k) in enumerate(dt.items()):
        ans += token_sent[j]+" "
    sumsize = sent_tokenize(ans)
    
    summary = ""
    for i in range(n):
        summary += sumsize[i]+" "
    return summary


#text = read_file(filepath)
def generate_summary(text, no_of_lines):
    #text = read_file(text1)
    #print("\nOriginal text : \n", text)
    token_sent=sent_tokenize(text)
    cleaned_text = data_cleaning(text)
    
    unique_words = unique_word(cleaned_text)
    #print(len(unique_words))
    doc_info = get_doc(cleaned_text)
    freq_dict_list = create_freq_matrix(cleaned_text)
    
    tf_values = tf_matrix(doc_info, unique_words, freq_dict_list)
    idf_values = idf_matrix(doc_info, unique_words, freq_dict_list)
    tfidf_values = TFIDF_scores(doc_info, unique_words, tf_values,idf_values)
    
    cosine_sim_mat = cosine_matrix(token_sent, tfidf_values)
    scores = textrank(cosine_sim_mat)
    
    summary = sent_rank(token_sent, scores,no_of_lines)
    return summary






#ans = generate_summary("msft.txt",7)
#print("\n\nSummarized text : \n",ans)
