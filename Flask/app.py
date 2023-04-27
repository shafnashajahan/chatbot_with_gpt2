from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods = ['GET', 'POST'])
def home():
    #return render_template('index.html')
    return render_template('index1.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)