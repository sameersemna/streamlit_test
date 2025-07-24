import streamlit as st
import numpy as np
from sklearn.preprocessing import Normalizer
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import json
import ast
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import traceback

MAX_SEQUENCE_LENGTH = 60
VOCAB_LIMIT = 40000

try:
    stopwords.words('english') # Attempt to access to check if already downloaded
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt') # Attempt to access to check if already downloaded
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab') # Attempt to access to check if already downloaded
except LookupError:
    nltk.download('punkt_tab')

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


def get_pad_sequence(df):
    df['comb_tokens_fr'] = df['comb_tokens_fr'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    all_tokens_filtered = []
    for token_list in df['comb_tokens_fr']:
        all_tokens_filtered.extend(token_list)
    # Count token frequencies
    token_counter = Counter(all_tokens_filtered)
    # Keep only top N most frequent tokens
    most_common_tokens = dict(token_counter.most_common(VOCAB_LIMIT))
    vocab_filtered = {word: i+1 for i, word in enumerate(most_common_tokens.keys())}
    vocab_size_filtered = len(vocab_filtered) + 1

    def tokens_to_sequences_filtered(token_list):
        return [vocab_filtered[token] for token in token_list if token in vocab_filtered]
        
    sequences = [tokens_to_sequences_filtered(tokens) for tokens in df['comb_tokens_fr']]
    print(sequences)
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

@st.cache_data
def load_keras_model_and_predict(model_path: str, data_df: pd.DataFrame):
    """
    Loads a Keras model from a specified path and uses it to make predictions
    on a pandas DataFrame.

    Args:
        model_path (str): The file path to the saved Keras model (e.g., 'my_model.h5').
        data_df (pd.DataFrame): The DataFrame containing the input data for prediction.
                                Ensure its columns/structure match the model's expected input.

    Returns:
        np.ndarray: The predictions made by the model.
    """
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'.")
        return None

    try:
        # 1. Load the Keras model
        st.write(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        # model.summary() # Print model summary for verification
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Ensure the model file is valid and TensorFlow is correctly installed.")
        return None

    # 2. Prepare the DataFrame for prediction
    # Keras models typically expect NumPy arrays as input.
    # The shape and data type must match what the model was trained on.
    # For a tabular model, often directly converting the DataFrame to a NumPy array is sufficient.
    # If your model expects specific preprocessing (e.g., scaling, normalization,
    # specific column order, one-hot encoding, or different input for images/text),
    # apply that preprocessing here before converting to NumPy.

    # print('***************************************************************')
    dt = get_pad_sequence(data_df[0])
    # dt = data_df['comb_tokens_fr'].iloc[0]
    # print(dt)
    # st.table(dt)

    # Example: Assuming data_df contains numerical features suitable for direct input
    # Convert DataFrame to a NumPy array
    # dt = to_categorical(dt)
    # input_data_np = data_df.values # Or select specific columns: data_df[feature_columns].values
    input_data_np = np.array(dt) 
    # st.table(input_data_np)

    # Reshape if necessary (e.g., for recurrent layers expecting (samples, timesteps, features))
    # if model.input_shape[1:] != input_data_np.shape[1:]:
    #     st.warning(f"Input data shape {input_data_np.shape} might not match model input shape {model.input_shape}. Reshaping might be needed.")
    #     # Example for 1D input if model expects (None, num_features) but data is (num_samples,)
    #     if len(input_data_np.shape) == 1 and len(model.input_shape) == 2:
    #         input_data_np = input_data_np.reshape(-1, 1) # Reshape to (samples, 1)

    st.write(f"Input data shape for prediction: {input_data_np.shape}")
    if len(data_df) == 2:
        # input_data_np = np.array(input_data_np[0])
        input_image_np = np.array([data_df[1]])
        st.write(f"Input image shape for prediction: {input_image_np.shape}")

    # 3. Make predictions
    st.write("Making predictions...")
    try:
        if len(data_df) == 2:
            predictions = model.predict([input_data_np, input_image_np])
        else:
            predictions = model.predict(input_data_np)
        st.success("Predictions generated successfully!")
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Check if input data shape and type match the model's expectations.")

        traceback.print_exc()
        return None

### JANEKs functions
def display_html_file(file_path, height=None):
    """
    Display an HTML file in Streamlit.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    html_content = html_content.replace('background: #6c757d;', 'background: transparent;')

    if height is None:
        st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.components.v1.html(html_content, height=height)

def load_keras_model(model_path):
    """Load a complete Keras model saved with model.save()"""
    try:
        model = keras.models.load_model(model_path)
        st.success(f"Complete model loaded from: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading complete model: {e}")
        return None
    
from PIL import Image

def show_pdf_page(pdf_path, pnum=2):
    """Show page 2 of a PDF as an image in streamlit"""
    doc = fitz.open(pdf_path)
    page = doc[pnum-1]  # Page 2 (0-indexed)
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5x zoom
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    st.image(img)
    doc.close()

def numpy_to_json(numpy_array: np.ndarray, indent: int = None) -> str:
    """
    Converts a NumPy array to a JSON string.

    Args:
        numpy_array (np.ndarray): The NumPy array to convert.
        indent (int, optional): If not None, JSON array elements will be pretty-printed
                                with that indent level. Defaults to None (compact output).

    Returns:
        str: A JSON string representation of the NumPy array.
    """
    if not isinstance(numpy_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    # Convert the NumPy array to a Python list
    # .tolist() handles multi-dimensional arrays correctly,
    # converting them into nested lists.
    python_list = numpy_array.tolist()

    # Convert the Python list to a JSON string
    json_string = json.dumps(python_list, indent=indent)

    return json_string
