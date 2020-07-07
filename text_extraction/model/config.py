import transformers
import tokenizers
MAX_LEN = 192
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 5
BERT_PATH = r"E:\personal_project\text_extraction\model\roberta-base"
MODEL_PATH = r"E:\personal_project\text_extraction\model.bin"
TRAINING_FILE = r"E:\personal_project\text_extraction\model\train.csv"
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{BERT_PATH}/vocab.json", 
    merges_file=f"{BERT_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)