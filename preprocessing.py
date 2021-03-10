import os
from os import listdir
from os.path import isfile, join
from io import open
import numpy as np
import classes

#Given the path of the articles folder, return array of arrays of words (array of articles, where each article is an array of words).
#Returns the array of articles, the array of summaries and the dictionary
def read_files(path):

  onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

  dic = classes.Dictionary()

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
              if(line.rstrip()):
                summary = summary + line.rstrip() + ' . '
                flag = 0

    articles.append(article)
    summaries.append(summary)

  return articles, summaries, dic




#Given the articles (or summaries) strings and a dictionary, return the arrays of words converted to indexes
def prepare_data(articles, dic):

  articles_idx = []

  for article in articles:

    article_idx = []
    words = article.rstrip().split()

    for word in words:

      article_idx.append(dic.word2idx[word])

    articles_idx.append(article_idx)

  return articles_idx


#Pads the articles (or summaries) to the length of longest article (or summary)
def zero_pad(art_idx):

  longest = 0
  padded_articles = []

  #Length of longest article
  for art in art_idx:
    if(len(art) > longest):
      longest = len(art)

  for art in art_idx:
    padded_article = np.zeros(longest, np.long)
    padded_article[:len(art)] = art
    padded_articles.append(padded_article)

  return padded_articles
