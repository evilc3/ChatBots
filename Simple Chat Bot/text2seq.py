from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer




def get_sequences(data):

    #tokenization 

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(data)

    sequences = tokenizer.texts_to_sequences(data)

    #calculate the max_len for padding 
    max_len = max([len(x) for x in sequences])

    print(f'max_len = {max_len}')

    sequences = pad_sequences(sequences,maxlen=max_len,padding = "post")


    return sequences,tokenizer,max_len


def get_classes(labels):

    lb = LabelBinarizer()

    lb.fit(labels)

    return lb.transform(labels),lb
    
def get_intent(classes,bl):

    return bl.inverse_transform(classes)




def get_processed_input(input,tok,max_len):

    X  = tok.texts_to_sequences([input])
    
    X  = pad_sequences(X,maxlen = max_len,padding="post")

    return X
