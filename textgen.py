from flask import Flask,request,render_template
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

app = Flask(__name__)

def load_object(file_path):
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

tokenizer = load_object('artifacts/tokenizer.pkl')
model = load_model('artifacts/model.keras')

def predict_fn(input_text,next_words):
    for _ in range(next_words):
        tkn_list = tokenizer.texts_to_sequences([input_text])[0]
        tkn_list = pad_sequences([tkn_list],maxlen = 78, padding='pre')
        predicted = np.argmax(model.predict(tkn_list),axis = -1)
        output_text = ''
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_text = word
                break
        input_text += " " + output_text
    return input_text

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def do_prediction():
    a = request.form.get('inputText')
    b = int(request.form.get('wordCount'))
    
    msg = predict_fn(a,b)
    
    return render_template('index.html',text=msg)

if __name__ == "__main__":
    app.run(debug=True)
    