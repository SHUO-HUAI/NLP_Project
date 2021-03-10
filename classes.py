# The dictionary contains:
#<pad> (index 0)
#<unk> (index 1)
# Then we will add also the words <start>, <end>...

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

        self.add_word('<pad>')#0
        self.add_word('<unk>')#1

    def add_word(self, word):
        # word = word.lower()
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
