from gensim.models.word2vec import Word2Vec # Собственно модель.
from gensim.models import KeyedVectors
import numpy as np
import pymorphy2
import re
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
#from simple_elmo import ElmoModel
#model = ElmoModel()
PATH_TO_ELMO='E:\MIEM\IOD_seminars\model.bin'
#model.load(PATH_TO_ELMO)

class Normalizer:
    conv_pos = {'ADJF':'ADJ', 'ADJS':'ADJ', 'ADV':'ADV', 'NOUN':'NOUN', 
            'VERB':'VERB', 'PRTF':'ADJ', 'PRTS':'ADJ', 'GRND':'VERB'}
    tmp_dict = {} # Кеш значимых слов.
    nones = {} # Кеш незначимых слов.
    morph = pymorphy2.MorphAnalyzer()
    def normalizeText(self,text):
        tokens = re.findall('[A-Za-zА-Яа-яЁё]+\-[A-Za-zА-Яа-яЁё]+|[A-Za-zА-Яа-яЁё]+', text)
        words = []
        for t in tokens:
            # Если токен уже был закеширован, быстро возьмем результат из него.
            if t in self.tmp_dict.keys():
                words.append(self.tmp_dict[t])
            # Аналогично, если он в кеше незначимых слов.
            elif t in self.nones.keys():
                pass
            # Слово еще не встретилось, будем проводить медленный морфологический анализ.
            else:
                pv = self.morph.parse(t)
                if pv[0].tag.POS != None:
                    if pv[0].tag.POS in self.conv_pos.keys():
                        word = pv[0].normal_form
                        # Отправляем слово в результат, ...
                        words.append(word + '_' + str(pv[0].tag.POS))
                        # ... и кешируем результат его разбора.
                        self.tmp_dict[t] = word
                    else:
                        # Для незначимых слов можно даже ничего не хранить. Лишь бы потом не обращаться к морфологии.
                        self.nones[t] = ""                
        return words
test1=Normalizer()   
model_w2v = KeyedVectors.load_word2vec_format(PATH_TO_ELMO, binary=True)

with open('data_SAPR.json') as f:
    d = json.load(f)
index2word_set = list(model_w2v.index_to_key)

#print(model.get_elmo_vectors(d))

    
def text_to_vec(text):
    text_vec = np.zeros((model_w2v.vector_size,), dtype="float32")
    n_words = 0
    testV=test1.normalizeText(text)
    for word in testV:
        if word in index2word_set:
            n_words = n_words + 1
            text_vec = np.add(text_vec, model_w2v[word]) 

    if n_words != 0:
        text_vec /= n_words
    return text_vec

w2v_vectors = [text_to_vec(text) for text in tqdm(d)]
np.save('lab4/ArticlesVectors', w2v_vectors)
