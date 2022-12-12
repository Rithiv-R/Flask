from flask import Flask,request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from translate import Translator
import requests
import easyocr


def myvalue(url):
    response = requests.get(url)
    if response.status_code==200:
        with open("bus.jpeg","wb") as f:
            f.write(response.content)
    else:
        print(response.status_code)

def reads():
    read = easyocr.Reader(['ta','en'])
    bounds = read.readtext('bus.jpeg')
    return bounds[0][1]

def translate(name):
    trans = Translator(from_lang='ta',to_lang='en')
    result = trans.translate(name)
    return result

app = Flask(__name__)
api = Api(app)

@app.route('/service',methods=['GET'])
def find():
    if request.method=='GET':
        print(request.args['url'])
        myvalue(str(request.args['url']))
        name = reads()
        print(name)
        english_name = translate(name)
        print(english_name)
        return english_name

if __name__ == '__main__':
    app.run()