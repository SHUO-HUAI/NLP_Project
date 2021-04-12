import os
from os import listdir
from os.path import isfile, join
from io import open
import numpy as np
import time
import torch
import classes
import subprocess
import pickle
from classes import Dictionary, CountDictionary

# according to https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py
dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
SENTENCE_START = '<SOS>'
SENTENCE_END = '<EOS>'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  #
MAX_DIC_LEN = 50000
MAX_ART_LEN = 400
MAX_SUM_LEN = 100

os.environ['CLASSPATH'] = './stanford-corenlp/stanford-corenlp-4.2.0.jar'


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image
    # captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return article, abstract


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which "
            "has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


# Given the path of the articles folder, return array of arrays of words (array of articles, where each article is an
# array of words). Returns the array of articles, the array of summaries and the dictionary
# def read_files(path):
#     onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
#     dic = Dictionary()
#     articles = []
#     summaries = []
#     for f in onlyfiles:
#         with open(join(path, f), 'r', encoding="utf8") as myfile:
#             article = ''
#             summary = ''
#             flag = 0
#             for line in myfile:
#                 if '@highlight' in line:
#                     flag = 1
#                 else:
#                     words = line.rstrip().split()  # + ['<eos>']
#                     for word in words:
#                         dic.add_word(word)
#                     if flag == 0:
#                         if line.rstrip():
#                             article = article + line.rstrip() + ' '
#                     else:
#                         if line.rstrip():
#                             summary = summary + line.rstrip() + ' . '
#                             flag = 0
#         articles.append(article)
#         summaries.append(summary)
#     return articles, summaries, dic


def read_files(stories_path, tokenized_path):
    if type(tokenized_path) is not list:
        stories_path = [stories_path]
        tokenized_path = [tokenized_path]
    dic = Dictionary()
    articles = []
    summaries = []

    for i in range(len(tokenized_path)):
        need_token = not os.path.exists(tokenized_path[i])
        if need_token:
            os.mkdir(tokenized_path[i])
            tokenize_stories(stories_path[i], tokenized_path[i])

        stories = os.listdir(tokenized_path[i])
        for story in stories:
            article, abstract = get_art_abs(os.path.join(tokenized_path[i], story))
            articles.append(article)
            summaries.append(abstract)

            art_tokens = article.split(' ')
            abs_tokens = abstract.split(' ')
            abs_tokens = [t for t in abs_tokens if
                          t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if t != ""]  # remove empty
            for token in tokens:
                dic.add_word(token)

    return articles, summaries, dic


def prepare_dictionary(train_path, dic_out_path):
    dic = Dictionary()
    count_dic = CountDictionary()
    # i = 0
    print(len(dic.word2idx))
    with open(train_path, 'r', encoding="utf8") as f:
        for art_name in f:
            article, summary = get_art_abs(art_name.strip())

            art_tokens = article.split(' ')
            art_tokens = [t.strip() for t in art_tokens]  # strip
            art_tokens = [t for t in art_tokens if t != ""]  # remove empty
            sum_tokens = summary.split(' ')
            sum_tokens = [t.strip() for t in sum_tokens]  # strip
            sum_tokens = [t for t in sum_tokens if t != ""]  # remove empty

            for token in art_tokens:
                count_dic.add_word(token)

            for token in sum_tokens:
                count_dic.add_word(token)
            #
            # i += 1
            # if (i == 2000):
            #     break

    print(len(count_dic.word2count))
    # print(count_dic.word2count)

    sorted_items = sorted(count_dic.word2count.items(), key=lambda item: item[1], reverse=True)
    # print(sorted_items[:100])

    j = 0
    for word_count in sorted_items:
        # print(word_count[0])
        dic.add_word(word_count[0])
        j += 1
        if (j == MAX_DIC_LEN):
            break
    print(j)
    print(len(dic.word2idx))

    with open(dic_out_path, 'wb+') as output:
        pickle.dump(dic, output, pickle.HIGHEST_PROTOCOL)
    return dic


def prepare_train_art_sum(train_path, dic_out_path, out_path):
    articles_idx = []
    summaries_idx = []

    dic = Dictionary()
    i = 0

    with open(train_path, 'r', encoding="utf8") as f:
        for art_name in f:
            article, summary = get_art_abs(art_name.strip())

            art_tokens = article.split(' ')
            art_tokens = [t.strip() for t in art_tokens]  # strip
            art_tokens = [t for t in art_tokens if t != ""]  # remove empty
            sum_tokens = summary.split(' ')
            sum_tokens = [t.strip() for t in sum_tokens]  # strip
            sum_tokens = [t for t in sum_tokens if t != ""]  # remove empty

            art_idx = []
            sum_idx = []

            for token in art_tokens:
                dic.add_word(token)
                art_idx.append(dic.word2idx[token])

            for token in sum_tokens:
                dic.add_word(token)
                sum_idx.append(dic.word2idx[token])

            articles_idx.append(art_idx)
            summaries_idx.append(sum_idx)

            if i > 100:
                break
            i = i + 1

        padded_articles = zero_pad(articles_idx)
        padded_summaries = zero_pad(summaries_idx)

        padded_train = np.transpose(np.array([padded_articles, padded_summaries], dtype=object))

        with open(dic_out_path, 'wb+') as output:
            pickle.dump(dic, output, pickle.HIGHEST_PROTOCOL)

        if not os.path.exists(out_path):
            os.mkdir(out_path)

        with open(os.path.join(out_path, 'train_set'), 'wb+') as output:
            pickle.dump(padded_train, output, pickle.HIGHEST_PROTOCOL)

    return dic


def prepare_art_sum(path, out_path, dic):
    articles_idx = []
    summaries_idx = []
    # i = 0

    with open(path, 'r', encoding="utf8") as f:
        for art_name in f:
            article, summary = get_art_abs(art_name.strip())

            oov_dic = {}
            oov_idx = len(dic.word2idx)

            art_tokens = article.split(' ')
            art_tokens = [t.strip() for t in art_tokens]  # strip
            art_tokens = [t for t in art_tokens if t != ""]  # remove empty
            sum_tokens = summary.split(' ')
            sum_tokens = [t.strip() for t in sum_tokens]  # strip
            sum_tokens = [t for t in sum_tokens if t != ""]  # remove empty

            art_idx = []
            sum_idx = []

            art_len = 0
            sum_len = 0

            for token in art_tokens:

                if token in dic.word2idx.keys():
                    art_idx.append(dic.word2idx[token])

                else:
                    if token in oov_dic:
                        art_idx.append(oov_dic[token])

                    else:
                        art_idx.append(oov_idx)
                        oov_dic[token] = oov_idx
                        oov_idx += 1

                art_len += 1
                if (art_len >= MAX_ART_LEN):
                    break

            for token in sum_tokens:
                if token in dic.word2idx.keys():
                    sum_idx.append(dic.word2idx[token])

                else:
                    if token in oov_dic:
                        sum_idx.append(oov_dic[token])

                    else:
                        sum_idx.append(oov_idx)
                        oov_dic[token] = oov_idx
                        oov_idx += 1

                sum_len += 1
                if (sum_len >= MAX_SUM_LEN):
                    break

            # print(oov_idx)

            articles_idx.append(art_idx)
            summaries_idx.append(sum_idx)

            # i = i + 1
            # if i > 10000:
            #     break

        padded_articles = zero_pad(articles_idx)
        padded_summaries = zero_pad(summaries_idx)

        # print(len(padded_articles[0]))
        # print(len(padded_summaries[0]))

        padded_data = np.transpose(np.array([padded_articles, padded_summaries], dtype=object))

        with open(out_path, 'wb+') as output:
            pickle.dump(padded_data, output, pickle.HIGHEST_PROTOCOL)


# Given the articles strings and a dictionary, return the arrays of words converted to indexes
def prepare_data(articles, dic):
    articles_idx = []
    for article in articles:
        article_idx = []
        # words = article.rstrip().split()
        art_tokens = article.split(' ')
        tokens = [t.strip() for t in art_tokens]  # strip
        tokens = [t for t in tokens if t != ""]  # remove empty
        oov_idx = len(dic.idx2word)
        for word in tokens:
            if word in dic.word2idx.keys():
                article_idx.append(dic.word2idx[word])
            else:
                article_idx.append(oov_idx)
                oov_idx += 1
        articles_idx.append(article_idx)
    return articles_idx


# Same as prepare_data but this is for summary so will add <SOS> at beginning and <EOS> at end
def prepare_summary(summaries, dic):
    summaries_idx = []
    for summary in summaries:
        abs_tokens = summary.split(' ')
        tokens = [t.strip() for t in abs_tokens]  # strip
        tokens = [t for t in tokens if t != ""]  # remove empty
        summary_idx = []
        for word in tokens:
            summary_idx.append(dic.word2idx[word])
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
        if art_idx[i] != 0:
            k += 1
    return k


def get_unked(articles, dic):
    articles_idx = []
    for article in articles:
        article_idx = []
        for word in article:
            if word >= len(dic.word2idx):
                article_idx.append(dic.word2idx['<UNK>'])
            else:
                article_idx.append(word)
        articles_idx.append(article_idx)
    return torch.tensor(articles_idx)
