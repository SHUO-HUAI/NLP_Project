import os
import sys
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  #

# download and save in the root folder, ignored by gitignore due to large size
os.environ['CLASSPATH'] = './stanford-corenlp/stanford-corenlp-4.2.0.jar'


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
    # print line[-1]
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


def write_to_bin(stories_dir, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a
    out_file. """

    stories = os.listdir(stories_dir)
    num_stories = len(stories)
    print(num_stories)

    if makevocab:
        vocab_counter = collections.Counter()

    for story in stories:
        article, abstract = get_art_abs(os.path.join(stories_dir, story))
        print(article)
        print(abstract)
        # Write the vocab to file, if applicable
        if makevocab:
            art_tokens = article.split(' ')
            abs_tokens = abstract.split(' ')
            abs_tokens = [t for t in abs_tokens if
                          t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if t != ""]  # remove empty
            # vocab_counter.update(tokens)
            for t in tokens:
                print(t)
            # print(tokens)


    # # write vocab to file
    # if makevocab:
    #     print("Writing vocab file...")
    #     with open(os.path.join(out_file, "vocab"), 'w') as writer:
    #         for word, count in vocab_counter.most_common(200000):
    #             writer.write(word + ' ' + str(count) + '\n')
    #     print("Finished writing vocab file")


tokenize_stories('../data', './data_tttt')
write_to_bin('./data_tttt', './data_eeee', True)
