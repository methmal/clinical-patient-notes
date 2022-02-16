from transformers import AutoTokenizer

num_workers = 2
scheduler = 'cosine'
batch_scheduler = True
num_cycles = 0.5
n_fold = 5
model = "/remote/vgatr_ur_temp/data/pretrained_models/deberta-base/large"
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.save_pretrained("../tokenizer")
max_len=466
apex = True
gradient_accumulation_steps =1
max_grad_norm = 1000
batch_scheduler = True
print_freq =100
device = "cuda"
# LOGGER = utils.get_logger()
# OUTPUT_DIR = "../output/train"
OUTPUT_DIR = "/remote/vgatr_ur_temp/data/clinical-patient-notes/output/"
batch_size = 12
encoder_lr = 2e-5
decoder_lr = 2e-5
weight_decay = 0.01
eps = 1e-6
betas = (0.9 ,0.999)
epochs = 5
seed = 42
trn_fold = [0, 1, 2, 3, 4]
train = True
num_warmup_steps =0
fc_dropout=0.2
