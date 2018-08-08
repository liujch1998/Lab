import parsing
import model

sentences = parsing.load_data()
lang = parsing.Lang(sentences)
inputs = [lang.sentence_to_indices(sentence) for sentence in sentences]
targets = [lang.sentence_to_indices(sentence) for sentence in sentences]
encoder, decoder = model.init_model(lang)
model.train_all(inputs, targets, encoder, decoder)
outputs = model.eval_all(inputs[:10], targets[:10], encoder, decoder)
for i in range(10):
    print('< ' + lang.indices_to_sentence(inputs[i]))
    print('> ' + lang.indices_to_sentence(outputs[i]))
