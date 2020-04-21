# Models
There are 3 word embeddings models:\
Fasttext, Word2Vec, GloVe\
\
A word embedding model contains a list of numbers representing each word in its vocabulary. Words with similar meaning have close values.\
Each model usually is described with a name, the dimension of the representation vector and the quantity of words in its vocabulary.\
For example:\
glove.twitter.27B.25d.txt\
It's a glove model, using tweeter source for its scraping algorithm. It has 27 billion words and each word is represented by an array of 25 dimensions.\
\
Usually the models are either .txt or .bin\
.txt models contains a csv with the first element being the word and the next X values the values of their dimensions\
\
## Using the algorithms\
\
The sentence similarity will require a model to work with, so the first thing to be done is to load the model.\
First download the model that you want to use.\
\
Fasttext models: https://fasttext.cc/docs/en/english-vectors.html\
Word2Vec models: https://code.google.com/archive/p/word2vec/\
GloVe models: https://nlp.stanford.edu/projects/glove/\
\
Then use the corresponding algorithm to load the chosen model:\
load_glove(model_directory, p_model) to load a GloVe model\
load_word2vec(model_directory, p_model) to load a Word2Vec model\
load_fasttext(model_directory, p_model) to load a fasttext model\
\
model_directory is the name of the directory where the model is downloaded, for example\
"D:/Datasets/GloVe/" (Last "/" is optional)\
p_model is the enum equivalent to the model you want to use. Let's say you've downloaded "glove.twitter.27B.zip", extracted the files and wants to use the "glove.twitter.27B.25d.txt" as a model\
You need to first get the equivalent Enum.\
Open the file "models_enum" and check which enum is equivalent to that file (You can use ctrl + F and put the name of the file)\
You will find the following line inside "glove_models" enum:\
TWITTER_27B_25D = 7         # glove.twitter.27B.25d.txt     (glove.twitter.27B.zip)\
So, to use this, your "p_model" must be: glove_models.TWITTER_27B_25D\
\
Wrapping it all up, the code you have to run to load this model is:\
model = load_glove("D:/Datasets/GloVe/", glove_models.TWITTER_27B_25D)\
\
Considering that your "glove.twitter.27B.25d.txt" file is on the directory D:/Datasets/GloVe/\
\
Once the model is loaded, you need to create the sentences you want to compare.\
s1 = 'President greets the press in Chicago'\
s2 = 'Obama speaks in Illinois'\
\
Then, just run the function "sentence_similarity_by_cosine_distance":\
sentence_similarity_by_cosine_distance(s1, s2, model)\
\
This code is written in "Test.py", using the smaller dataset, since it's only for testing.\
