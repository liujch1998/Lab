# encoding: utf-8

import parsing
import model

sentences = parsing.load_data()
lang = parsing.Lang(sentences)
inputs = [lang.sentence_to_indices(sentence) for sentence in sentences]
targets = [lang.sentence_to_indices(sentence)[::-1] for sentence in sentences]
encoder, decoder = model.init_model(lang)
model.train_all(inputs, targets, encoder, decoder)

sentences_test = [
    u'山抹微云天连衰草画角声断谯门',
    u'暂停征棹聊共引离尊',
    u'多少蓬莱旧事空回首烟霭纷纷',
    u'斜阳外寒鸦万点流水绕孤村',
    u'销魂',
    u'当此际香囊暗解罗带轻分',
    u'谩赢得青楼薄幸名存',
    u'此去何时见也襟袖上空惹啼痕',
    u'伤情处高城望断灯火已黄昏',
]
inputs_test = [lang.sentence_to_indices(sentence) for sentence in sentences_test]
outputs_test = model.eval_all(inputs_test, encoder, decoder)
for i in range(9):
    print('< ' + lang.indices_to_sentence(inputs_test[i]))
    print('> ' + lang.indices_to_sentence(outputs_test[i][::-1]))
