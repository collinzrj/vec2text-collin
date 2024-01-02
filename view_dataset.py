from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel
from vec2text.models.config import InversionConfig
from transformers import DataCollatorForWholeWordMask, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
import datasets
from tqdm import tqdm


if __name__ == '__main__':
    dataset = datasets.load_from_disk('~/entity_private_prompts_dataset')
    original_dataset = datasets.load_dataset('jxm/private_prompts')
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    json_dataset = []
    prompt_value_dict = {}
    for idx, data in tqdm(enumerate(original_dataset['train'])):
        prompt_value_dict[data['prompt']] = data['value']
    for idx, data in enumerate(dataset['train']):
        print(idx, data)
        print(prompt_value_dict[data['suffix']])
        break