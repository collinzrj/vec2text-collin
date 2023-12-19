from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel, InversionMaskedLogitsModelEncoder
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
import logging



class CustomDataCollator(DataCollatorForWholeWordMask):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        res = super().torch_call(examples)
        logits_list = []
        for e in examples:
            logits_list.append(e['frozen_embeddings'])
        res['logits'] = torch.stack(logits_list)
        print(res['input_ids'])
        return res

if __name__ == '__main__':
    # config = InversionConfig.from_json_file('masked_config.json')
    config = InversionConfig.from_pretrained('test_save')
    model = InversionMaskedLogitsModelEncoder(config)
    dataset = datasets.load_from_disk('./logits_dataset_small/train')
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    new_dataset = []
    logging.info("Start")
    for data in dataset:
        sentence = t5_tokenizer.decode(token_ids=data['input_ids'])
        data['input_ids'] = roberta_tokenizer.encode(sentence)
        new_dataset.append(data)
    logging.info("End")
    collator = CustomDataCollator(tokenizer=roberta_tokenizer, mlm=True, mlm_probability=0.15)
    DATASET_SPLIT = int(len(new_dataset) * 0.8)
    train_dataset = new_dataset[:DATASET_SPLIT]
    val_dataset = new_dataset[DATASET_SPLIT:]
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=collator, batch_size=1)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, collate_fn=collator, batch_size=1)
    EPOCH_NUM = 10
    optimizer = Adam(model.parameters(), lr=5e-6)
    total_iterations = len(train_loader) * EPOCH_NUM
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=total_iterations)
    for epoch in range(EPOCH_NUM):
        model.train()
        losses = []
        for step, samples in enumerate(train_loader):
            optimizer.zero_grad()
            output: MaskedLMOutput = model(**samples)
            output.loss.backward()
            optimizer.step()
            lr_scheduler.step()
            losses.append(output.loss.detach().numpy())
            if (step % 100) == 0:
                losses = [loss for loss in losses if not np.isnan(loss)]
                print(epoch, step, np.average(losses), lr_scheduler.get_lr())
                losses = []
        model.eval()
        with torch.no_grad():
            losses = []
            for step, samples in enumerate(val_loader):
                output: MaskedLMOutput = model(**samples)
                losses.append(output.loss.detach().numpy())
            losses = [loss for loss in losses if not np.isnan(loss)] 
            print("val loss", np.average(losses))
        
        