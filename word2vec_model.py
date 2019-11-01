from gensim.models import word2vec
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('word2vec_corpus/text8')
model = word2vec.Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
model.save('word2vec_corpus/word2vec.model')
# model = word2vec.Word2Vec.load('word2vec_corpus/word2vec.model')
# print('0.17654265463352203' in model)