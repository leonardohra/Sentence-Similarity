
from models_enum import glove_models, word2vec_models, fasttext_models

import io
import re
import string
from unidecode import unidecode

import numpy as np
import gensim
import scipy
from nltk.corpus import stopwords
from nltk import word_tokenize

'''
    Before using this function, download the glove model you want to use.
    If you don't know in which zip file the following txt files are, check
    'models_enum.py'. There you will find a comment with the zip related to
    each txt file.
    https://nlp.stanford.edu/projects/glove/
    Usage example:
    You've downloaded the glove.6B.zip, and extracted it to C:/Glove Datasets
    You want to use glove.6B.200d.txt
    glove_embeddings_dict = load_glove('C:/Glove Datasets/', glove_models.WIKI_6B_200D)
'''
def load_glove(model_directory, p_model = glove_models.WIKI_6B_50D):
    glove_files = {
        glove_models.WIKI_6B_50D: 'glove.6B.50d.txt',
        glove_models.WIKI_6B_100D: 'glove.6B.100d.txt',
        glove_models.WIKI_6B_200D: 'glove.6B.200d.txt',
        glove_models.WIKI_6B_300D: 'glove.6B.300d.txt',
        glove_models.CRAWL_42B_300D: 'glove.42B.300d.txt',
        glove_models.CRAWL_840B_300D: 'glove.840B.300d.txt',
        glove_models.TWITTER_27B_25D: 'glove.twitter.27B.25d.txt',
        glove_models.TWITTER_27B_50D: 'glove.twitter.27B.50d.txt',
        glove_models.TWITTER_27B_100D: 'glove.twitter.27B.100d.txt',
        glove_models.TWITTER_27B_200D: 'glove.twitter.27B.200d.txt'
    }

    embeddings_dict = {}
    model_path = model_directory
    model_path += glove_files[p_model] if model_path[-1] == '/' else '/' + glove_files[p_model]

    with open(model_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    return embeddings_dict

'''
    Before using this function, download the word2vec model you want to use.
    If you don't know in which zip file the following txt files are, check
    'models_enum.py'. There you will find a comment with the zip related to
    each txt file.
    https://code.google.com/archive/p/word2vec/
    Usage example:
    You've downloaded the GoogleNews-vectors-negative300.bin.gz, and extracted it to C:/Word2Vec Datasets
    You want to use GoogleNews-vectors-negative300.bin
    word2vec_model = load_word2vec('C:/Word2Vec Datasets/', word2vec_models.GOOGLE_NEWS_VECTOR_NEGATIVE_300)
'''
def load_word2vec(model_directory, p_model = word2vec_models.GOOGLE_NEWS_VECTOR_NEGATIVE_300):
    word2vec_files = {
        word2vec_models.GOOGLE_NEWS_VECTOR_NEGATIVE_300: 'GoogleNews-vectors-negative300.bin',
        word2vec_models.KNOWLEDGE_VECTORS_SKIPGRAM_1000: 'knowledge-vectors-skipgram1000.bin',
        word2vec_models.KNOWLEDGE_VECTORS_SKIPGRAM_1000_EN: 'knowledge-vectors-skipgram1000-en.bin'
    }

    model_path = model_directory
    model_path += word2vec_files[p_model] if model_path[-1] == '/' else '/' + word2vec_files[p_model]

    model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=True)

    return model

'''
    Before using this function, download the fasttext model you want to use.
    If you don't know in which zip file the following txt files are, check
    'models_enum.py'. There you will find a comment with the zip related to
    each txt file.
    https://fasttext.cc/docs/en/english-vectors.html
    Usage example:
    You've downloaded the crawl-300d-2M.vec.zip, and extracted it to C:/Fasttext Datasets
    You want to use crawl-300d-2M.vec
    fasttext_embeddings_dict = load_fasttext('C:/Fasttext Datasets/', CRAWL_300D_2M_VEC)
'''
def load_fasttext(model_directory, p_model = fasttext_models.CRAWL_300D_2M_VEC):
    fasttext_files = {
        fasttext_models.CRAWL_300D_2M_VEC: 'crawl-300d-2M.vec',
        fasttext_models.CRAWL_300D_2M_SUBWORD_BIN: 'crawl-300d-2M-subword.bin',
        fasttext_models.CRAWL_300D_2M_SUBWORD_VEC: 'crawl-300d-2M-subword.vec',
        fasttext_models.WIKI_NEWS_300D_1M_VEC: 'wiki-news-300d-1M.vec',
        fasttext_models.WIKI_NEWS_300D_1M_SUBWORD_VEC: 'wiki-news-300d-1M-subword.vec'
    }

    model_path = model_directory
    model_path += fasttext_files[p_model] if model_path[-1] == '/' else '/' + fasttext_files[p_model]

    with (io.open(model_path, 'r', encoding='utf-8', newline='\n', errors='ignore')) as fin:
        n, d = map(int, fin.readline().split())
        embeddings_dict = {}

        for line in fin:
            tokens = line.rstrip().split(' ')
            embeddings_dict[tokens[0]] = map(float, tokens[1:])

    return embeddings_dict

'''
    This function will clean the text and return tokens, for example:
    'The man is on the train, but he is late.' (input)
    ['late', 'train', 'man'] (output)
    All words are in lowercase
    Stopwords were removed
    Repeated words were removed
'''
def preprocess(text):
    # Remove non ascii characters
    text = unidecode(text)

    # Convert to lower case and split
    words = text.lower()
    words_tk = word_tokenize(words)

    # Remove stopwords
    stopword_set = set(stopwords.words("english")  + list(string.punctuation))
    cleaned_words = []

    for w in words_tk:
        if w not in stopword_set and w not in cleaned_words:
            cleaned_words.append(w)

    return cleaned_words

'''
Each word is represented by an x dimensional array (depending on the model)
To calculate the cosine distance, we will get the mean of each of these dimensions
related to each word.
For example:
'Beautiful' = [0.5, 0.2, 0.1]
'House' = [0.5, 0.6, 0.8]
The array that represents the sentence "Beautiful House" is: [ 0.5, 0.4, 0.45 ]
'''
def vectorize_sentence(s1, model):
    return np.mean([model[word] for word in preprocess(s1)],axis=0)

'''
In the method below, we calculate the cosine distance of both arrays
'''
def cosine_distance(v1, v2):
    cosine = scipy.spatial.distance.cosine(v1, v2)
    return cosine

'''
Uses cosine distance to calculate sentence similarity
'''
def sentence_similarity_by_cosine_distance(s1, s2, model):
    v1 = vectorize_sentence(s1, model)
    v2 = vectorize_sentence(s2, model)
    cosine = cosine_distance(v1, v2)

    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
    return cosine
