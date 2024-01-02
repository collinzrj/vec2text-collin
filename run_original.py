from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel, InversionMaskedLogitsModelEncoder
from vec2text.models.config import InversionConfig
from transformers import DataCollatorForWholeWordMask, AutoTokenizer, DataCollatorForLanguageModeling
from transformers.modeling_outputs import MaskedLMOutput
import datasets
from torch.utils.data import DataLoader
from typing import List, Union, Any, Dict
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from datetime import datetime
from tqdm import tqdm

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

class EntityMaskCollator(DataCollatorForLanguageModeling):
    def __init__(self):
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        prompt_value_dict = {}
        original_dataset = datasets.load_dataset('jxm/private_prompts')
        for idx, data in tqdm(enumerate(original_dataset['train'])):
            prompt_value_dict[data['prompt']] = (data['value'], data['field'], data['source'])
        self.prompt_value_dict = prompt_value_dict
        # self.super()

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        res = {'input_ids': [], 'labels': [], 'logits': []}
        max_batch_length = -1
        for idx in range(len(examples)):
            prompt = examples[idx]['suffix']
            prompt_tokens = self.roberta_tokenizer.encode(prompt)
            max_batch_length = max(max_batch_length, len(prompt_tokens))
        for idx in range(len(examples)):
            prompt = examples[idx]['suffix']
            entity, _, _ = self.prompt_value_dict[examples[idx]['suffix']]
            prompt_tokens = self.roberta_tokenizer.encode(prompt)
            prompt_enc = self.roberta_tokenizer(prompt)
            start_char_idx = prompt.find(entity)
            end_char_idx = start_char_idx + len(entity) - 1
            start_token = prompt_enc.char_to_token(start_char_idx)
            end_token = prompt_enc.char_to_token(end_char_idx)
            # pad with -1, which is token for <pad>
            prompt_tokens = prompt_tokens + [1 for _ in range(max_batch_length - len(prompt_tokens))]
            prompt_tokens = torch.tensor(prompt_tokens)
            labels = torch.clone(prompt_tokens)
            tokens_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
            tokens_mask[start_token:end_token] = True
            prompt_tokens[tokens_mask] = 50264
            labels[tokens_mask == False] = -100
            res['input_ids'].append(prompt_tokens)
            res['labels'].append(labels)
            res['logits'].append(examples[idx]['frozen_embeddings'])
        for key in res.keys():
            res[key] = torch.stack(res[key])
        return res

if __name__ == '__main__':
    if False:
        model = InversionMaskedLogitsModel.from_pretrained('logits_inversion_decoder_no_logits_not_causal_model_save_loss_1p85').to(device)
        model.use_logits = False
    elif True:
        model = InversionMaskedLogitsModel.from_pretrained('logits_inversion_decoder_use_logits2').to(device)
        model.use_logits = True
    else:
        config = InversionConfig.from_pretrained('test_save')
        model = InversionMaskedLogitsModelEncoder(config).to(device)
    if False:
        val_dataset = datasets.load_from_disk('/root/entity_private_prompts_dataset/validation')
        collator = EntityMaskCollator()
    else:
        val_dataset = datasets.load_from_disk('./logits_dataset/validation')
        roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        collator = CustomDataCollator(tokenizer=roberta_tokenizer, mlm=True, mlm_probability=0.15)
    for _ in range(10):
        val_loader = DataLoader(dataset=val_dataset, shuffle=True, collate_fn=collator, batch_size=10)
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
        
        
