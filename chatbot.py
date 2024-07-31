import pickle 
import random 
import numpy as np 
import json
import nltk 
from nltk.stem import WordNetLemmatizer 
from keras.models import load_model 

lemmatizer = WordNetLemmatizer()

intents = json.load(open('intents.json', 'r'))

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence): 
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words 

def bag_of_words(sentence): 
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words): 
            if word == w:
                bag[i] = 1
    
    return np.array(bag)

def predict_class(sentence, ERROR_THRESHOLD=0.25):
    # Process the input sentence into a bag of words
    bow = bag_of_words(sentence)
    
    # Ensure bow has the correct shape
    bow = np.array(bow).reshape(1, -1)
    
    # Predict the class
    res = model.predict(bow)[0]  # Get the prediction (first element)
    
    # Filter out predictions below the ERROR_THRESHOLD
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability (high to low)
    result.sort(key=lambda x: x[1], reverse=True)
    
    return result

def get_response(intents_list, intents_json):
    if not intents_list:  # Handle case where no intent exceeds the threshold
        return "I'm not sure what you mean. Can you please clarify?"
    
    list_of_intents = intents_json['intents']
    tag = classes[intents_list[0][0]]  # Get the tag with the highest probability
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break 
            
    return result 

print("Great Bot is running")

while True:
    message = input()
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
