import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import TweetModel
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn import model_selection
from sklearn.model_selection import KFold
results=[]
def run():
    dfx=pd.read_csv(config.TRAINING_FILE)
    df_train,df_valid=model_selection.train_test_split(
    dfx,
    test_size=0.1,
    random_state=42
)
    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )
    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )
    device=torch.device('cuda')
    model_config=transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states=True
    model=TweetModel(conf=model_config)
    model.to(device)
    
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
        )
    for epoch in range(3):
        train_fn(train_data_loader, model, optimizer,device, scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        results.append(jaccard)
        print(f"Jaccard Score = {jaccard}")
        torch.save(model.state_dict(),f"MODEL_PATH")