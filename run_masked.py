from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel
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

device = torch.device('cuda')

class CustomDataCollator(DataCollatorForWholeWordMask):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.15):
        self.t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        for idx in range(len(examples)):
            sentence = self.t5_tokenizer.decode(examples[idx]['input_ids'])
            new_tokens = self.roberta_tokenizer.encode(sentence)
            examples[idx]['input_ids'] = torch.tensor(new_tokens)
        res = super().torch_call(examples)
        logits_list = []
        for e in examples:
            logits_list.append(e['frozen_embeddings'])
        res['logits'] = torch.stack(logits_list).to(device)
        res['input_ids'] = res['input_ids'].to(device)
        return res

def find_slice(seq, subseq):
    n = len(seq)
    m = len(subseq)
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            return (i, i + m)
    return (-1, -1)


class EntityMaskCollator:
    def __init__(self):
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        prompt_value_dict = {}
        original_dataset = datasets.load_dataset('jxm/private_prompts')
        for idx, data in tqdm(enumerate(original_dataset['train'])):
            print(data)
            prompt_value_dict[data['prompt']] = (data['value'], data['field'], data['source'])
        self.prompt_value_dict = prompt_value_dict

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        res = {'input_ids': [], 'labels': [], 'logits': []}
        for idx in range(len(examples)):
            prompt = examples[idx]['suffix']
            entity = self.prompt_value_dict[examples[idx]['suffix']]
            prompt_tokens = self.roberta_tokenizer.encode(prompt)
            entity_tokens = self.roberta_tokenizer.encode(entity)
            start_idx, end_idx = find_slice(prompt_tokens, entity_tokens)
            if start_idx == -1:
                raise LookupError
            prompt_tokens = torch.tensor(prompt_tokens)
            labels = torch.clone(prompt_tokens)
            tokens_mask = torch.zeros_like(prompt_tokens, dtype=torch.bool)
            tokens_mask[start_idx:end_idx] = True
            prompt_tokens[tokens_mask] = 50264
            labels[tokens_mask == False] = -100
            res['input_ids'].append(prompt_tokens)
            res['labels'].append(labels)
            res['logits'] = examples[idx]['frozen_embeddings']
        for key in res.keys():
            res[key] = torch.stack(res[key])
        return res

if __name__ == '__main__':
    # config = InversionConfig.from_pretrained('test_save')
    # config = InversionConfig.from_json_file('masked_config.json')
    # model = InversionMaskedLogitsModel(config).to(device)
    # TRAIN_NO_LOGITS = False
    if False:
        SAVE_FILE = 'logits_inversion_decoder_no_logits_not_causal_model'
        if True:
            model = InversionMaskedLogitsModel.from_pretrained(SAVE_FILE).to(device)
        else:
            config = InversionConfig.from_json_file('masked_config.json')
            model = InversionMaskedLogitsModel(config).to(device)
        model.use_logits = False
    else:
        SAVE_FILE = 'logits_inversion_decoder_use_logits'
        model = InversionMaskedLogitsModel.from_pretrained(SAVE_FILE).to(device)
        model.use_logits = True
    train_dataset = datasets.load_from_disk('./logits_dataset/train')
    val_dataset = datasets.load_from_disk('./logits_dataset/validation')
    # t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    collator = CustomDataCollator(tokenizer=roberta_tokenizer, mlm=True, mlm_probability=0.15)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, collate_fn=collator, batch_size=96)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, collate_fn=collator, batch_size=100)
    EPOCH_NUM = 100
    optimizer = Adam(model.parameters(), lr=5e-6)
    total_iterations = len(train_loader) * EPOCH_NUM
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=total_iterations)
    with open('run_masked.log', 'w') as f:
        f.write("")
    for epoch in range(EPOCH_NUM):
        losses = []
        for step, samples in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            output: MaskedLMOutput = model(**samples)
            output.loss.backward()
            optimizer.step()
            lr_scheduler.step()
            losses.append(output.loss.detach().cpu().numpy())
            if (step % 100) == 0:
                losses = [loss for loss in losses if not np.isnan(loss)]
                with open(f'run_masked_epoch_{EPOCH_NUM}.log', 'a') as f:
                    f.write(f"{datetime.now().strftime('%H:%M:%S')} {epoch} {step} {np.average(losses)} {lr_scheduler.get_lr()}\n")
                print(f"{datetime.now().strftime('%H:%M:%S')} {epoch} {step} {np.average(losses)} {lr_scheduler.get_lr()}")
                losses = []
            if (step % 1000) == 0:
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
                model.save_pretrained(SAVE_FILE)
        
        
