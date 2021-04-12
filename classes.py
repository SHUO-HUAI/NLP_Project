# The dictionary contains:
# <pad> (index 0)
# <unk> (index 1)
# Then we will add also the words <start>, <end>...

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

        self.add_word('<PAD>')  # 0
        self.add_word('<UNK>')  # 1
        self.add_word('<SOS>')  # 2
        self.add_word('<EOS>')  # 3

    def add_word(self, word):
        # word = word.lower()
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
        


class CountDictionary(object):
    def __init__(self):
        self.word2count = {}

    def add_word(self, word):
        # word = word.lower()
        if word not in self.word2count.keys():
            self.word2count[word] = 1
        
        else:
            self.word2count[word] += 1

    def __len__(self):
        return len(self.idx2word)
