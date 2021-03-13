import os
from os import listdir
from os.path import isfile, join
from io import open
import numpy as np
import torch
import classes
from classes import Dictionary


# Given the path of the articles folder, return array of arrays of words (array of articles, where each article is an
# array of words). Returns the array of articles, the array of summaries and the dictionary
def read_files(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    dic = Dictionary()

    articles = []
    summaries = []
    for f in onlyfiles:

        with open(join(path, f), 'r', encoding="utf8") as myfile:
            article = ''
            summary = ''
            flag = 0
            for line in myfile:

                if '@highlight' in line:
                    flag = 1

                else:

                    words = line.rstrip().split() + ['<eos>']
                    for word in words:
                        dic.add_word(word)

                    if flag == 0:
                        if line.rstrip():
                            article = article + line.rstrip() + ' '
                    else:
                        if line.rstrip():
                            summary = summary + line.rstrip() + ' . '
                            flag = 0

        articles.append(article)
        summaries.append(summary)

    return articles, summaries, dic


# Given the articles strings and a dictionary, return the arrays of words converted to indexes
def prepare_data(articles, dic):
    articles_idx = []

    for article in articles:

        article_idx = []
        words = article.rstrip().split()

        for word in words:
            article_idx.append(dic.word2idx[word])

        articles_idx.append(article_idx)

    return articles_idx
    
    
# Same as prepare_data but this is for summary so will add <SOS> at beginning and <EOS> at end
def prepare_summary(summaries, dic):
    summaries_idx = []

    for summary in summaries:

        summary_idx = []
        summary_idx.append(dic.word2idx['<SOS>'])
        words = summary.rstrip().split()

        for word in words:
            summary_idx.append(dic.word2idx[word])
        
        summary_idx.append(dic.word2idx['<EOS>'])

        summaries_idx.append(summary_idx)

    return summaries_idx


# Pads the articles (or summaries) to the length of longest article (or summary)
def zero_pad(art_idx):
    longest = 0
    padded_articles = []

    # Length of longest article
    for art in art_idx:
        if len(art) > longest:
            longest = len(art)


    for art in art_idx:
        padded_article = np.zeros(longest, np.long)
        padded_article[:len(art)] = art
        padded_articles.append(padded_article)
    

    return padded_articles
    
    
def remove_pad(art_idx):

    k = 0
    
    for i in range(len(art_idx)):
        
        if(art_idx[i] != 0):
            k += 1
        
    return k
    


#Given an article or a list of articles (in words, not indexes) returns the indexes of the words, and replaces new words with the index corresponding to '<unk>'

def get_unked(articles, dic):

  articles_idx = []

  for article in articles:

    article_idx = []
    words = article.rstrip().split()

    for word in words:

      if word not in dic.word2idx:
          article_idx.append(dic.word2idx['<unk>'])

      else:
          article_idx.append(dic.word2idx[word])

    articles_idx.append(article_idx)

  return articles_idx
