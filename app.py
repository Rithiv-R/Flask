from flask import Flask,request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer
import torch.nn

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
nlp = spacy.load('en_core_web_sm')
model1 = SentenceTransformer('distilbert-base-nli-mean-tokens')


def summarization(answer):

    stopwords = list(STOP_WORDS)

    doc = nlp(answer)

    tokens = [token.text for token in doc]


    punctuation1 = punctuation + '\n'

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation1:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency

    sentence_tokens = [sent for sent in doc.sents]

    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    select_length = int(len(sentence_tokens)*0.5)

    summary = nlargest(select_length, sentence_scores, key = sentence_scores.get)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    return summary

def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

def rephrasedsimilarity(sanswer,keyvalue):

    arr = []
    sentences = nlp(sanswer)
    for val in sentences.sents:
        if(str(val)!='\n '):
            arr.append(str(val))

    rephrased = []
    for i in arr:
        s = get_response(i, 10)
        rephrased.append(s)
    sentokens = nlp(keyvalue)

    key = []
    for sent in sentokens.sents:
        if(str(sent)!='\n '):
            key.append(str(sent))

    keylength = len(key)
    sentence_embeddings = model1.encode(key)
    myembeddings = [ model1.encode(i) for i in rephrased ]

    cos_com = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    keyscript = torch.from_numpy(sentence_embeddings)

    best = []
    for key_val in keyscript:
        maxofall = []
        for i in myembeddings:
            answerscript = torch.from_numpy(i)
            maxoften = []
            for j in answerscript:
                maxoften.append(cos_com(key_val,j))
            maxofall.append(max(maxoften))
        best.append(max(maxofall))
    
    final_mark = 0
    for i in best:
        final_mark+=float(i)*((1/keylength)*100)

    return final_mark


app = Flask(__name__)
api = Api(app)

CORS(app)

@app.route('/service',methods=['GET'])
def marks():
    if request.method=='GET':
        summary = summarization(str(request.args['answer']))
        marks = rephrasedsimilarity(summary,str(request.args['key']))
        return {'summary':summary,'marks':marks}

if __name__ == '__main__':
    app.run()