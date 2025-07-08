import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from nltk.corpus import stopwords
import pandas as pd
from nltk.corpus import stopwords
from functions import preprocess_sentence, print_closest, find_closest, get_closest, compare

df = pd.read_csv("MovieReview.csv")
# display(df.head())
print(df.shape)

df = df.drop('sentiment', axis=1)

stop_words = stopwords.words('english')

df.review = df.review.apply(lambda x :preprocess_sentence(x))

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df.review)
word2idx = tokenizer.word_index
idx2word = tokenizer.index_word
vocab_size = tokenizer.num_words

embedding_dim = 300
max_sequence_length = 256 # Example: Max 256 words per input sequence

model = Sequential([
    # 1. Embedding Layer: This is crucial for word embeddings.
    #    It maps integer-encoded words to dense vectors.
    #    The weights for this layer are what 'word2vec.h5' likely contains.
    #    input_length is essential for defining the input shape of the model.
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),

    # 2. Global Average Pooling Layer: Reduces the sequence of embeddings to a single vector.
    GlobalAveragePooling1D(),

    # 3. Dense Layer: Your output layer for classification or other tasks.
    #    The output size (vocab_size here) depends on your specific task.
    Dense(vocab_size, activation='softmax')
])

# Explicitly build the model to finalize its architecture and create weights.
# This is crucial, especially if the model isn't built implicitly by data input.
model.build(input_shape=(None, max_sequence_length)) # (None, max_sequence_length) for batch_size, sequence_length

# Now, load the weights. The model's layers are now defined and built.
try:
    model.load_weights("word2vec.h5")
    st.success("Model weights loaded successfully!")
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.info("Please ensure 'word2vec.h5' exists and the model architecture (vocab_size, embedding_dim, max_sequence_length, and layer types/order) precisely matches the saved weights.")

# You can now use your model
st.write("Model Summary:")
model.summary()

# Example of using the model (replace with your actual application logic)
# For instance, if you want to get the embedding vectors:
vectors = model.layers[0].trainable_weights[0].numpy()
st.write(f"Shape of loaded embedding vectors: {vectors.shape}")

# Further Streamlit app logic would go here
st.title("Movie Streamlit App (with Word2Vec)")

vectors = model.layers[0].trainable_weights[0].numpy()

focus_word = 'zombie'
st.write(f"print_closest: {focus_word}")
# print_closest(focus_word, word2idx, vectors, idx2word)
focus_word_closest = get_closest(focus_word, word2idx, vectors, idx2word, 20)
st.table(focus_word_closest)

focus_word = 'warrior'
st.write(f"print_closest: {focus_word}")
focus_word_closest = get_closest(focus_word, word2idx, vectors, idx2word, 20)
st.table(focus_word_closest)

index_word1='zombie'
index_word2='warrior'
index_word3='vampire'
st.write(f"compare: {index_word1}, {index_word2}, {index_word3}")
comparison = compare(word2idx[index_word1], word2idx[index_word2], word2idx[index_word3], vectors, 1)
st.table(comparison)
