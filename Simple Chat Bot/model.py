import tensorflow as tf
from tensorflow.keras.layers import Dense,Embedding,Dropout,Flatten
from tensorflow.keras.models import Sequential



def get_model(vocab_size,emb_dim,input_len,n_classes):

    tf.random.set_seed(101)
    

    return Sequential([
        Embedding(input_dim=vocab_size,output_dim = emb_dim,input_length=input_len),
        Dense(128,activation='relu'),
        Dense(64,activation = 'relu'),
        Dropout(0.3),
        Flatten(),
        Dense(n_classes,activation='softmax')

    ])





