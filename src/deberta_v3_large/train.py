import itertools
import torch.nn as nn
import gc
import engine
import time
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch
from transformers import logging
import pandas as pd
import utils
import ast
from sklearn.model_selection import GroupKFold , StratifiedKFold , KFold
from config import CFG
from torch.utils.data import DataLoader
from dataset import TrainDataset
from model import CustomModel
from torch.optim import AdamW
logging.set_verbosity_warning()



def load_data():
    train = pd.read_csv('/remote/vgatr_ur_temp/data/clinical-patient-notes/input/train.csv')
    train['annotation'] = train['annotation'].apply(ast.literal_eval)
    train['location'] = train['location'].apply(ast.literal_eval)
    features = pd.read_csv('/remote/vgatr_ur_temp/data/clinical-patient-notes/input/features.csv')
    def preprocess_features(features):
        features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
        return features
    features = preprocess_features(features)
    patient_notes = pd.read_csv('/remote/vgatr_ur_temp/data/clinical-patient-notes/input/patient_notes.csv')

    train = train.merge(features, on=['feature_num', 'case_num'], how='left')
    train = train.merge(patient_notes, on=['pn_num', 'case_num'], how='left')

    train = utils.data_clean(train)
    train['annotation_length'] = train['annotation'].apply(len)


    Fold = GroupKFold(n_splits=CFG.n_fold)
    groups = train['pn_num'].values
    for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)

    return train
def train_loop(folds, fold , LOGGER):
    

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_texts = valid_folds['pn_history'].values
    valid_labels = utils.create_labels_for_scoring(valid_folds)
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, CFG.OUTPUT_DIR+'config.pth')
    model.to(CFG.device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler=='cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.
num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_score = 0.


    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = engine.train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)

        # eval
        avg_val_loss, predictions = engine.valid_fn(valid_loader, model, criterion, CFG.device)
        predictions = predictions.reshape((len(valid_folds), CFG.max_len))

        # scoring
        char_probs = utils.get_char_probs(valid_texts, predictions, CFG.tokenizer)
        results = utils.get_results(char_probs, th=0.5)
        preds = utils.get_predictions(results)
        score = utils.get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')


        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        CFG.OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")


    predictions = torch.load(CFG.OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[i for i in range(CFG.max_len)]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
def get_result(oof_df):
    labels = utils.create_labels_for_scoring(oof_df)
    predictions = oof_df[[i for i in range(CFG.max_len)]].values
    char_probs = utils.get_char_probs(oof_df['pn_history'].values, predictions, CFG.tokenizer)
    results = utils.get_results(char_probs, th=0.5)
    preds = utils.get_predictions(results)
    score = utils.get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}')

if __name__ == "__main__" :
    torch.cuda.memory_summary(device=None, abbreviated=False)
    LOGGER = utils.get_logger()
    train = load_data()
    utils.get_maxlen()
    utils.seed_everything()

    #training
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(train , fold , LOGGER)
            oof_df = pd.concat([oof_df , _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
    oof_df = oof_df.reset_index(drop=True)
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    oof_df.to_pickle(CFG.OUTPUT_DIR+'oof_df.pkl')


