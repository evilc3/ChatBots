'''
The intents are our classes eg. greet, info etc.
Data Structure 
Input Data        Output Data

Text1                intent1            

each text has only one intent , each intent can have n number of text (question associated with it.)
Once intent is detected.
'''

import json 


def get_data(filename):


    file = open(filename)

    dataset  = json.load(file)

    response = {}
    requests = []
    labels = []

    for intent in dataset['intents']:


            # keys : tags (Y_value) , patterns (X_value) , response (ChatBots Response)
            for request_text in intent['patterns']:
                requests.append(request_text)    
                

            response[intent['tag']] = intent['responses']


            labels.extend([*[intent['tag']] * len(intent['patterns'])])

    

    print(f' loaded {len(requests)} patterns from {filename}')

    print(f' loaded {len(labels)} intents from {filename}')

    return requests,response,labels


if __name__ == "__main__":

    print(get_data('data.json'))      


