from model import config
import torch
import flask
import time
from flask import Flask
from flask import request,render_template
from model.model import TweetModel
from model.dataset import TweetDataset
import functools
import torch.nn as nn
from tqdm import tqdm
import joblib
import transformers
from model.engine import calculate_jaccard_score
app = flask.Flask(__name__,template_folder='tamplates')
MODEL = None
device = "cuda"
def prediction(tweet,sentiment,selected_text):
    final_output = []
    test_dataset = TweetDataset(
        tweet=tweet,
        sentiment=sentiment,
        selected_text=selected_text
        )
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=1
        )
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            outputs_start, outputs_end = MODEL(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                _, output_sentence = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                    )
                final_output.append(output_sentence)
    return final_output
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def get_data():
    context=request.form['context']
    sentiment=request.form['sentiment']
    selected_text=request.form['context']
    sel_txt=prediction(context,sentiment,selected_text)
    return render_template('test.html',sel_txt=sel_txt)


if __name__ == "__main__":
    model_config=transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states=True
    MODEL = TweetModel(conf=model_config)
    MODEL = torch.nn.DataParallel(MODEL)
    try:
        model_state_dict = MODEL.module.state_dict()
    except AttributeError:
        model_state_dict = MODEL.state_dict()
    MODEL.to(device)
    MODEL.eval()
    app.run()