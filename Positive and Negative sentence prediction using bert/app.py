from model import config
import torch
import flask
import time
from flask import Flask
from flask import request,render_template
from model.model import BERTBaseUncased
import functools
import torch.nn as nn
import joblib
import transformers
import torch.nn as nn
app = flask.Flask(__name__,template_folder='tamplates')
MODEL = None
DEVICE = "cuda"
PREDICTION_DICT = dict()
def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def get_data():
    Sentence=request.form['Sentence']
    sentiment=sentence_prediction(Sentence)
    return render_template('index.html',sentiment=sentiment)
    
if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL = torch.nn.DataParallel(MODEL)
    try:
        model_state_dict = MODEL.module.state_dict()
    except AttributeError:
        model_state_dict = MODEL.state_dict()
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()
