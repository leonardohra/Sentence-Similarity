from algorithms import *

def main():
    model_directory = 'D:/Datasets/GloVe/' # Change this to the directory your model is in
    p_model = glove_models.WIKI_6B_50D

    s1 = 'President greets the press in Chicago'
    s2 = 'Obama speaks in Illinois'
    model = load_glove(model_directory, p_model)
    
    cosine = sentence_similarity_by_cosine_distance(s1, s2, model)

if __name__ == "__main__":
    main()
