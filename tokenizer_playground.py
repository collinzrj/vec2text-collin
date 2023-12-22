from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel
from vec2text.models.config import InversionConfig
from .transformers import DataCollatorForWholeWordMask, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
import datasets


if __name__ == '__main__':
    dataset = datasets.load_from_disk('./logits_dataset/validation')
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    while True:
        prompt = input("Please input prompt:\n")
        print(roberta_tokenizer.encode(prompt))