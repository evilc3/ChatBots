from DataLoader import get_data
from text2seq import get_classes,get_sequences,get_processed_input,get_intent
from model import get_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pickle as pk
import numpy as np

def trainModel(filename):

    req,res,labels = get_data(filename)

    print('data loaded')

    Y , lb = get_classes(labels)

    print('labels processed')

    X,tok,max_len = get_sequences(req)

    print('text data processed')

    vocab_size = len(tok.word_index) + 1

    emb_dim = 32

    n_classes = len(lb.classes_)

    model = get_model(vocab_size,emb_dim,max_len,n_classes)


    print(model.summary())
    model.compile(loss = "categorical_crossentropy",optimizer= Adam(learning_rate=1e-3),metrics = ['accuracy'])
    model.fit(X,Y,epochs = 100)

    model.save("models/simple_bot.h5")

    print('model saved..')

    #also need to save the tokenzier,label_binirizer

    pk.dump(tok,open('models/tokenizer','wb'))
    pk.dump(lb,open('models/label_bin','wb'))
    pk.dump(max_len,open("models/parameters",'wb'))
    pk.dump(res,open("models/res",'wb'))
    


def inference(input,model,tok,bl,max_len):
   
   
    res = pk.load(open("models/res",'rb'))

    X = get_processed_input(input,tok,max_len)

    print(X.shape)
    
    intent = model.predict(X)

    print('class score:',intent)

    intent = get_intent(intent,bl)[0]

    print('intent : ' , intent)

    responces = res[intent]

    bot_responce = np.random.choice(responces,1)

    print('Bot:',bot_responce[0])






if __name__ == "__main__":

    op = int(input("Enter:\n1. to train bot.\n2. test bot.\n"))


    if op == 1:
        trainModel('data.json')

    else:
        #load the model 
        trained_model =   tf.keras.models.load_model('models/simple_bot.h5')
        #load the tokenizer 
        tok  = pk.load(open("models/tokenizer",'rb'))

        #load the label binirier 
        lb  = pk.load(open("models/label_bin",'rb'))

        #load the max_len 
        max_len  = pk.load(open("models/parameters",'rb'))

        print('bot starting .............')

        text = "."
        while text != "quit":
            text = input("Enter message:")
            inference(text,trained_model,tok,lb,max_len)
    









