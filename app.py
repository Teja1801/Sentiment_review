from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

with open('model/lr.pkl', 'rb') as file:
    model = joblib.load(file)

with open('model/vectorizer.pkl', 'rb') as file:
    vectorizer = joblib.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    text = request.form['text']
    text_vector = vectorizer.transform([text])
    sentiment = model.predict(text_vector)
    if text=='':
        return render_template('index.html', text='Please Enter Text')
    else:
        return render_template('index.html', valid=sentiment,text=text)
    
if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)