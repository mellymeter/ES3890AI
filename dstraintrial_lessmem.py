import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import random
import logging
import os
import torch
import torch.distributed
import transformers
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Sequence
from sklearn.model_selection import train_test_split
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from datasets import Dataset, load_dataset

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=".training_deep")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    per_device_train_batch_size: int = field(default=1)  # Reduced batch size to 1
    gradient_accumulation_steps: int = field(default=8)  # Accumulate gradients over 8 steps
    fp16: bool = field(default=False)  # Enable mixed precision training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def build_instruction_prompt(context: str, response: str):
    return '''
You are an AI that speaks like the character Anya from "Spy x Family". Respond to the following context as if you were Anya:
### Context:
{}
### Anya's Response:
{}
'''.format(context.strip(), response.strip()).lstrip()

def load_data_from_csv(csv_path: str, character_name: str) -> Tuple[Dataset, Dataset]:
    data = pd.read_csv(csv_path)
    contexted = []  # context window of size 7
    n = 7
    for i in data[data.name == character_name].index:
        if i < n:
            continue
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(data.line[j])
        contexted.append(row)

    columns = ['response', 'context'] + ['context/' + str(i) for i in range(n - 1)]
    df = pd.DataFrame.from_records(contexted, columns=columns)
    
    train_df, val_df = train_test_split(df, test_size=0.1)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    return train_dataset, val_dataset

def _tokenize_fn(strings: Sequence[str], tokenizer: AutoTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(text, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True)
        for text in strings
    ]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens)

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    sources = [
        build_instruction_prompt(context, response)
        for context, response in zip(examples['context'], examples['response'])
    ]
    targets = [f"{response}\n{EOT_TOKEN}" for response in examples['response']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.float16)  # Change to float16

    # Load datasets
    train_dataset, val_dataset = load_data_from_csv(data_args.data_path, 'Anya')

    # Tokenize datasets
    train_dataset = train_dataset.map(lambda examples: train_tokenize_function(examples, tokenizer), batched=True, remove_columns=["response", "context"] + [f"context/{i}" for i in range(6)])
    val_dataset = val_dataset.map(lambda examples: train_tokenize_function(examples, tokenizer), batched=True, remove_columns=["response", "context"] + [f"context/{i}" for i in range(6)])

    # Prepare data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_sampler = None

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Training
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

if __name__ == "__main__":
    clear_gpu_memory()
    train()