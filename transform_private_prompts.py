from vec2text.models.inversion_from_logits_masked import InversionMaskedLogitsModel
from vec2text.models.config import InversionConfig
from transformers import DataCollatorForWholeWordMask, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput
import datasets
from tqdm import tqdm


if __name__ == '__main__':
    dataset = datasets.load_from_disk('~/entity_private_prompts_dataset')
    original_dataset = datasets.load_dataset('jxm/private_prompts')
    json_dataset = []
    prompt_value_dict = {}
    for idx, data in tqdm(enumerate(original_dataset['train'])):
        print(data)
        prompt_value_dict[data['prompt']] = (data['value'], data['field'], data['source'])
    NUM_SHARDS = 10
    shard_datasets = []
    for rank in range(NUM_SHARDS):
        shard_dataset = dataset['train'].shard(num_shards=NUM_SHARDS, index=rank, contiguous=True)
        def map_fn(example):
            value, field, source = prompt_value_dict[example['suffix']]
            return {'entity': value, 'field': field, 'source': source}
        shard_dataset = shard_dataset.map(map_fn, num_proc=20)
        shard_datasets.append(shard_dataset)
    final_dataset: datasets.Dataset = datasets.concatenate_datasets(shard_datasets)
    final_dataset.save_to_disk('~/entity_private_prompts_dataset2', max_shard_size='2GB')