import transformers
import torch
from vec2text import analyze_utils

if __name__ == '__main__':
    experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
        "entity_prompt_model"
    )
