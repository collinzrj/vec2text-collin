from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel
from vec2text.models.config import InversionConfig
from transformers import DataCollatorForWholeWordMask, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
import datasets


if __name__ == '__main__':
    dataset = datasets.load_from_disk('./logits_dataset/validation2m')
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    json_dataset = []
    with open('prompts.txt', 'w') as f:
        for idx, data in enumerate(dataset):
            sentence = t5_tokenizer.decode(token_ids=data['input_ids'])
            f.write(f"{idx} {sentence}\n")