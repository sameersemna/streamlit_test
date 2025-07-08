import streamlit as st
import numpy as np
from sklearn.preprocessing import Normalizer
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- IMPORTANT: Download NLTK stopwords data if not already present ---
# This line ensures that the 'stopwords' corpus is available.
# It's good practice to place this where stopwords are first accessed.
try:
    stopwords.words('english') # Attempt to access to check if already downloaded
except LookupError:
    nltk.download('stopwords')

# Now you can safely use stopwords
stop_words = stopwords.words('english')

# --- Your existing functions from functions.py would go here ---
# (Assuming preprocess_sentence, print_closest, find_closest, get_closest, compare are defined here)

# Example placeholder for a function that might be in your functions.py
def nl_preprocess_sentence(sentence):
    """
    Example placeholder for sentence preprocessing.
    You would replace this with your actual preprocessing logic.
    """
    sentence = sentence.lower()
    # Remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Remove stopwords
    words = sentence.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def nl_print_closest(word, vectors, word_index, index_word, top_n=5):
    """Placeholder for printing closest words."""
    print(f"Closest words to '{word}':")
    # Actual logic would involve calculating cosine similarity etc.
    for i in range(top_n):
        print(f"  - Closest_Word_{i+1}")

def nl_find_closest(vector, vectors, index_word):
    """Placeholder for finding closest words."""
    # Actual logic would involve calculating cosine similarity etc.
    return ["closest_word_1", "closest_word_2"]

def nl_get_closest(word, vectors, word_index, index_word):
    """Placeholder for getting closest words."""
    if word in word_index:
        word_vector = vectors[word_index[word]]
        return find_closest(word_vector, vectors, index_word)
    return []

def nl_compare(sentence1, sentence2):
    """Placeholder for comparing sentences."""
    print(f"Comparing '{sentence1}' and '{sentence2}'")
    # Actual logic would involve vectorizing sentences and comparing them
    return "Comparison result placeholder"

def sentenceToData(tokens, WINDOW_SIZE):
    window = np.concatenate((np.arange(-WINDOW_SIZE,0),np.arange(1,WINDOW_SIZE+1)))
    X,Y=([],[])
    for word_index, word in enumerate(tokens) :
        if ((word_index - WINDOW_SIZE >= 0) and (word_index + WINDOW_SIZE <= len(tokens) - 1)) :
            X.append(word)
            Y.append([tokens[word_index-i] for i in window])
    return X, Y


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)

    # remove stopword
    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]
    return ' '.join(mots).strip()

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

@st.cache_data
def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

@st.cache_data
def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

@st.cache_data
def print_closest(word, word2idx, vectors, idx2word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        print(idx2word[index_word[1]]," -- ",index_word[0])

@st.cache_data
def get_closest(word, word2idx, vectors, idx2word, number=10):
    list_ret=[]
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        print(idx2word[index_word[1]]," -- ",index_word[0])
        list_ret.append([
            *index_word,
            idx2word[index_word[1]]
        ])
    return list_ret

