from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel
from vec2text.models.config import InversionConfig
from transformers import DataCollatorForWholeWordMask, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
import datasets
import numpy as np
import json


if __name__ == '__main__':
    dataset = datasets.load_from_disk('./logits_dataset/validation')
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    json_dataset = []
    dataset = list(dataset)
    masked_dict = {
        93: 'Explain the concept of <mask> like I am 5.</s>',
        101: 'Solve -30 = -74*j - <mask><mask> for j. Solve this problem.</s>',
        114: 'What is the recipe for <mask><mask>?</s>',
        224: 'Write a tweet that is <mask>.</s>',
        242: 'Explain the process of <mask><mask>.</s>',
        258: 'What are some of the major economic activities in the <mask><mask> region?</s>'
    }
    for idx in masked_dict.keys():
        data = dataset[idx]
        sample = {}
        sentence = t5_tokenizer.decode(token_ids=data['input_ids'])
        input_without_mask = np.array(roberta_tokenizer.encode(sentence))
        masked_sentence = masked_dict[idx]
        sample['input_ids'] = np.array(roberta_tokenizer.encode(masked_sentence))
        labels = input_without_mask.copy()
        labels[sample['input_ids'] != 50264] = -100
        sample['input_ids'] = sample['input_ids'].tolist()
        sample['logits'] = data['frozen_embeddings'].numpy().tolist()
        sample['labels'] = labels.tolist()
        json_dataset.append(sample)
    with open('privacy_dataset.json', 'w') as f:
        json.dump(json_dataset, f)