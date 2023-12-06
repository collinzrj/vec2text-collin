from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel
from vec2text.models.config import InversionConfig
from transformers import DataCollatorForWholeWordMask, AutoTokenizer
import datasets
from torch.utils.data import DataLoader
from typing import List, Union, Any, Dict
import torch




class CustomDataCollator(DataCollatorForWholeWordMask):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        res = super().torch_call(examples)
        logits_list = []
        for e in examples:
            logits_list.append(e['frozen_embeddings'])
        res['logits'] = torch.stack(logits_list)
        return res

if __name__ == '__main__':
    config = InversionConfig.from_json_file('test_config.json')
    model = InversionMaskedLogitsModel(config)
    dataset = datasets.load_from_disk('./logits_dataset/train')
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    new_dataset = []
    for data in dataset:
        sentence = t5_tokenizer.decode(token_ids=data['input_ids'])
        data['input_ids'] = roberta_tokenizer.encode(sentence)
        new_dataset.append(data)
    collator = CustomDataCollator(tokenizer=roberta_tokenizer, mlm=True, mlm_probability=0.15)
    data_loader = DataLoader(dataset=new_dataset, shuffle=True, collate_fn=collator, batch_size=1)
    for data in data_loader:
        # print(data)
        # print(data['logits'].shape)
        output = model(**data)
        print(output)
        break