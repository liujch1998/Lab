import json

puncs = [
    u'\uff0c', # ,
    u'\u3002', # .
    u'\u3001', # \
    u'\u2026', # ...
    u'\uff1a', # :
    u'\uff1f', # ?
    u'\uff01', # !
    u'\uff1b', # ;
]
purges = [
    u'\u0020', # space
    u'\u300a', # <
    u'\u300b', # >
    u'\u2fda', # a han
    u'\u30d4', # a han
    u'\u201c', # "
    u'\u201d', # "
    u'\u002e', # .
]
unks = [
    u'\u25a1', # blank square
    u'\uff3f', # _
]
UNK = u'\u0000'

def load_data ():
    poems = []
    for i in range(0, 22000, 1000):
        filename = 'chinese-poetry/ci/ci.song.' + str(i) + '.json'
        with open(filename, 'r') as f:
            poems += json.load(f)
    sentences = []
    for poem in poems:
        paragraphs = poem[u'paragraphs']
        for paragraph in paragraphs:
            is_valid = True
            sentence = u''
            for char in paragraph:
                if char in puncs or char in purges:
                    continue
                if char in unks:
                    sentence += UNK
                elif (ord(char) >= 0x4E00) and (ord(char) <= 0x9FFF):
                    sentence += char
                else:
                    is_valid = False
            if is_valid and sentence != u'':
                sentences.append(sentence)
    print('Loaded %d sentences' % len(sentences))
    return sentences

class Lang:

    def __init__ (self, sentences):
        self.han_cnt = 0
        self.han2index = {}
        self.index2han = {}
        for sentence in sentences:
            for han in sentence:
                self.add_han(han)

    def add_han (self, han):
        if han not in self.han2index:
            index = self.han_cnt
            self.han2index[han] = index
            self.index2han[index] = han
            self.han_cnt += 1

    def sentence_to_indices (self, sentence):
        return [self.han2index[han] for han in sentence]

    def indices_to_sentence (self, indices):
        return u''.join([self.index2han[index] for index in indices])
