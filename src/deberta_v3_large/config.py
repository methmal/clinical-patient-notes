from transformers.models.deberta_v2 import DebertaV2TokenizerFast

class CFG:
    wandb=False
    competition='NBME'
    _wandb_kernel='nakama'
    debug=False
    apex=True
    print_freq=100
    num_workers=4
    model="/remote/vgatr_ur_temp/data/pretrained_models/deberta-v3-large/"
    OUTPUT_DIR="/remote/vgatr_ur_temp/data/clinical-patient-notes/src/deberta_v3_large/output/"
    scheduler='linear' # ['linear', 'cosine']
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model)
    tokenizer.save_pretrained("../tokenizer")
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=5
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=12
    fc_dropout=0.2
    dropout=0.2
    high_dropout=0.5
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    n_fold=5
    trn_fold=[0,1,2,3,4]
    train=True
    device='cuda'
    
if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]
