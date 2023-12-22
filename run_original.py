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
from datetime import datetime

device = torch.device('cpu')

class CustomDataCollator(DataCollatorForWholeWordMask):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        self.t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        for idx in range(len(examples)):
            sentence = self.t5_tokenizer.decode(examples[idx]['input_ids'])
            # print(sentence)
            new_tokens = self.roberta_tokenizer.encode(sentence)
            examples[idx]['input_ids'] = torch.tensor(new_tokens)
        res = super().torch_call(examples)
        logits_list = []
        for e in examples:
            logits_list.append(e['frozen_embeddings'])
        res['logits'] = torch.stack(logits_list).to(device)
        res['input_ids'] = res['input_ids'].to(device)
        return res

if __name__ == '__main__':
    if False:
        model = InversionMaskedLogitsModel.from_pretrained('logits_inversion_decoder_no_logits_not_causal_model').to(device)
        model.use_logits = False
    elif True:
        model = InversionMaskedLogitsModel.from_pretrained('logits_inversion_decoder_use_logits').to(device)
        model.use_logits = True
    else:
        config = InversionConfig.from_pretrained('test_save')
        model = InversionMaskedLogitsModelEncoder(config).to(device)
    for _ in range(10):
        val_dataset = datasets.load_from_disk('./logits_dataset/validation')
        # t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
        roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        collator = CustomDataCollator(tokenizer=roberta_tokenizer, mlm=True, mlm_probability=0.15)
        val_loader = DataLoader(dataset=val_dataset, shuffle=True, collate_fn=collator, batch_size=100)
        model.eval()
        with torch.no_grad():
            losses = []
            accs = []
            for step, samples in enumerate(val_loader):
                output: MaskedLMOutput = model(**samples)
                # print(output.logits)
                # print(samples['labels'])
                losses.append(output.loss.detach().cpu().numpy())
                # mask_indices = torch.where(output.logits != 100)
                # print(mask_indices)
                num_label = torch.sum(samples['labels'] != -100, dim=1)
                predictions = torch.argmax(output.logits, dim=2).cpu()
                num_correct = torch.sum((samples['labels'] != -100) & (samples['labels'] == predictions), dim=1)
                acc = num_correct / num_label
                avg_acc = torch.mean(acc[torch.isnan(acc) == False]).detach().cpu().numpy()
                # print(acc)
                accs.append(avg_acc)
            losses = [loss for loss in losses if not np.isnan(loss)]
            accs = [acc for acc in accs if not np.isnan(acc)]
            print(f"{datetime.now().strftime('%H:%M:%S')}, val loss {np.average(losses)}, acc {np.average(accs)}")
        
        
