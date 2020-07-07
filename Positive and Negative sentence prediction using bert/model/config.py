import transformers
import tokenizers
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = r"E:\personal_project\Positive and Negative sentence prediction\model\bert-based-uncased"
MODEL_PATH = r"E:\personal_project\Positive and Negative sentence prediction\model.bin"
TRAINING_FILE = r"E:\personal_project\Positive and Negative sentence prediction\model\imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH,do_lower_case=True)