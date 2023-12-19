from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel
from vec2text.models.config import InversionConfig
from transformers import DataCollatorForWholeWordMask, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
import datasets
from torch.utils.data import DataLoader
from typing import List, Union, Any, Dict
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import json
from tqdm import tqdm
import pickle

if __name__ == '__main__':
    dataset = datasets.load_from_disk('./logits_dataset/train')
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    MAX_COUNT = 100000
    cnt = 0 
    print("Generating new dataset")
    FILE_CNT = 0
    new_data_list = []
    for idx, data in enumerate(tqdm(dataset)):
        sentence = t5_tokenizer.decode(token_ids=data['input_ids'])
        new_data = {}
        new_data['input_ids'] = roberta_tokenizer.encode(sentence)
        new_data['frozen_embeddings'] = data['frozen_embeddings'].numpy().tolist()
        new_data_list.append(new_data)
        if (idx + 1) % 20000 == 0:
            with open(f'logits_inversion_{FILE_CNT}.pkl', 'wb') as file:
                FILE_CNT += 1
                pickle.dump(new_data_list, file)
                new_data_list = []
    with open(f'logits_inversion_{FILE_CNT}.pkl', 'wb') as file:
        FILE_CNT += 1
        pickle.dump(new_data_list, file)
        new_data_list = []
    print("Finished generating new dataset")