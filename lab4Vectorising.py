from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import pymorphy2
import re
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

PATH_TO_ELMO='E:\MIEM\IOD_seminars\model.bin'

conv_pos = {'ADJF':'ADJ', 'ADJS':'ADJ', 'ADV':'ADV', 'NOUN':'NOUN', 
        'VERB':'VERB', 'PRTF':'ADJ', 'PRTS':'ADJ', 'GRND':'VERB'}
tmp_dict = {} # Кеш значимых слов.
nones = {} # Кеш незначимых слов.
morph = pymorphy2.MorphAnalyzer()
def normalizeText(text):
    tokens = re.findall('[A-Za-zА-Яа-яЁё]+\-[A-Za-zА-Яа-яЁё]+|[A-Za-zА-Яа-яЁё]+', text)
    words = []
    for t in tokens:
        if t in tmp_dict.keys():
            words.append(tmp_dict[t])
        elif t in nones.keys():
            pass
        else:
            pv = morph.parse(t)
            if pv[0].tag.POS != None:
                if pv[0].tag.POS in conv_pos.keys():
                    word = pv[0].normal_form
                    words.append(word + '_' + str(pv[0].tag.POS))
                    tmp_dict[t] = word
                else:
                    nones[t] = ""                
    return words 
model_w2v = KeyedVectors.load_word2vec_format(PATH_TO_ELMO, binary=True)

with open('lab4/data.json') as f:
    d = json.load(f)
    d1 = pd.DataFrame({'news':d})



index2word_set = set(model_w2v.index_to_key)

    
def text_to_vec(text):
    text_vec = np.zeros((model_w2v.vector_size,), dtype="float32")
    n_words = 0
    testV=normalizeText(text)
    for word in testV:
        if word in index2word_set:
            n_words = n_words + 1
            text_vec = np.add(text_vec, model_w2v[word]) 
    if n_words != 0:
        text_vec /= n_words
    return text_vec
w2v_vectors = [text_to_vec(text) for text in tqdm(d1['news'])]
np.save('lab4/ArticlesVectors', w2v_vectors)
