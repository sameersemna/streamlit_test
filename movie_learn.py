from nltk.corpus import stopwords
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from functions import preprocess_sentence, sentenceToData, unicode_to_ascii

df = pd.read_csv("MovieReview.csv")
# display(df.head())
print(df.shape)

df = df.drop('sentiment', axis=1)

# nltk.download()
stop_words = stopwords.words('english')

df.review = df.review.apply(lambda x :preprocess_sentence(x))
df.head()

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df.review)

word2idx = tokenizer.word_index
idx2word = tokenizer.index_word
vocab_size = tokenizer.num_words

WINDOW_SIZE = 5

X, Y = ([], [])
for review in df.review:
    for sentence in review.split("."):
        word_list = tokenizer.texts_to_sequences([sentence])[0]
        if len(word_list) >= WINDOW_SIZE:
            Y1, X1 = sentenceToData(word_list, WINDOW_SIZE//2)
            X.extend(X1)
            Y.extend(Y1)
    
X = np.array(X).astype(int)
y = np.array(Y).astype(int).reshape([-1,1])

embedding_dim = 300
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GlobalAveragePooling1D())
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size = 128, epochs=50)

model.save("word2vec.h5") 
model.save("word2vec.keras") 

# You are saving your model as an HDF5 file via 
# `model.save()` or `keras.saving.save_model(model)`. 
# This file format is considered legacy. 
# We recommend using instead the native Keras format, 
# e.g. `model.save('my_model.keras')` 
# or `keras.saving.save_model(model, 'my_model.keras')`. 

# Run: streamlit run movie_streamlit2.py 
