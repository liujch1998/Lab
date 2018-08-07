import json

poems = []
for i in range(0, 22000, 1000):
    filename = 'chinese-poetry/ci/ci.song.' + str(i) + '.json'
    with open(filename, 'r') as f:
        poems += json.load(f)
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
for poem in poems:
    paragraphs = poem[u'paragraphs']
    for paragraph in paragraphs:
        for char in paragraph:
            is_punc = any([char == punc for punc in puncs])
            is_purge = any([char == purge for purge in purges])
            is_unk = any([char == unk for unk in unks])
            is_han = (ord(char) >= 0x4E00) and (ord(char) <= 0x9FFF)
            if is_punc == False and is_purge == False and is_unk == False and is_han == False:
                print(char + ' ' + hex(ord(char)) + ' ' + paragraph)

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
                sentence += u'\u25a1'
            elif (ord(char) >= 0x4E00) and (ord(char) <= 0x9FFF):
                sentence += char
            else:
                is_valid = False
        if is_valid:
            sentences.append(sentence)

han_cnt = 0
han2index = {}
index2han = {}
for sentence in sentences:
    for han in sentence:
        if han not in han2index:
            index = han_cnt
            han2index[han] = index
            index2han[index] = han
            han_cnt += 1
print('Gathered %d Han characters' % han_cnt)
