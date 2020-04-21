from enum import Enum

'''
    All these files are found on: https://nlp.stanford.edu/projects/glove/
    The name of the file equivalent to the enum is on the left.
    The name of the zip where the file is found is on the right.
'''
class glove_models(Enum):
    WIKI_6B_50D = 1             # glove.6B.50d.txt              (glove.6B.zip)
    WIKI_6B_100D = 2            # glove.6B.100d.txt             (glove.6B.zip)
    WIKI_6B_200D = 3            # glove.6B.200d.txt             (glove.6B.zip)
    WIKI_6B_300D = 4            # glove.6B.300d.txt             (glove.6B.zip)
    CRAWL_42B_300D = 5          # glove.42B.300d.txt            (glove.42B.300d.zip)
    CRAWL_840B_300D = 6         # glove.840B.300d.txt           (glove.840B.300d.zip)
    TWITTER_27B_25D = 7         # glove.twitter.27B.25d.txt     (glove.twitter.27B.zip)
    TWITTER_27B_50D = 8         # glove.twitter.27B.50d.txt     (glove.twitter.27B.zip)
    TWITTER_27B_100D = 9        # glove.twitter.27B.100d.txt    (glove.twitter.27B.zip)
    TWITTER_27B_200D = 10       # glove.twitter.27B.200d.txt    (glove.twitter.27B.zip)

'''
    All these files are found on: https://code.google.com/archive/p/word2vec/
    The name of the file equivalent to the enum is on the left. *
    The name of the zip where the file is found is on the right.
    * One exception: if you want to use knowledge vectors skipgram 1000 - EN
    When you extract it, the file will probably be called "5" (only "5", god knows why)
    Please change the file name to "knowledge-vectors-skipgram1000-en.bin"
'''
class word2vec_models(Enum):
    GOOGLE_NEWS_VECTOR_NEGATIVE_300 = 1     # GoogleNews-vectors-negative300.bin        (GoogleNews-vectors-negative300.bin.gz)
    KNOWLEDGE_VECTORS_SKIPGRAM_1000 = 2     # knowledge-vectors-skipgram1000.bin        (freebase-vectors-skipgram1000.bin.gz)
    KNOWLEDGE_VECTORS_SKIPGRAM_1000_EN = 3  # knowledge-vectors-skipgram1000-en.bin*    (freebase-vectors-skipgram1000-en.bin.gz)

'''
    All these files are found on: https://fasttext.cc/docs/en/english-vectors.html
    The name of the file equivalent to the enum is on the left.
    The name of the zip where the file is found is on the right.
'''
class fasttext_models(Enum):
    CRAWL_300D_2M_VEC = 1               # crawl-300d-2M.vec                 (crawl-300d-2M.vec.zip)
    CRAWL_300D_2M_SUBWORD_BIN = 2       # crawl-300d-2M-subword.bin         (crawl-300d-2M-subword.zip)
    CRAWL_300D_2M_SUBWORD_VEC = 3       # crawl-300d-2M-subword.vec         (crawl-300d-2M-subword.zip)
    WIKI_NEWS_300D_1M_VEC = 4           # wiki-news-300d-1M.vec             (wiki-news-300d-1M.vec.zip)
    WIKI_NEWS_300D_1M_SUBWORD_VEC = 5   # wiki-news-300d-1M-subword.vec     (wiki-news-300d-1M-subword.vec.zip)

class embeddings(Enum):
    GLOVE = 1
    WORD2VEC = 2
    FASTTEXT = 3
