import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import torch

import accelerate.hooks
from datasets import load_dataset
from lm_eval import evaluator
from lm_eval.models.huggingface import AutoCausalLM, HuggingFaceAutoLM
from transformers import AutoTokenizer, PreTrainedModel

import nncf
import nncf.experimental
import nncf.experimental.torch.sparsify_activations
import nncf.experimental.torch.sparsify_activations.torch_backend
import nncf.quantization
import nncf.torch
import nncf.torch.quantization
import nncf.torch.quantization.quantize_model


class LMEvalModel(AutoCausalLM):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 512,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        device: Optional[Union[int, str]] = "cuda",
    ):
        super(HuggingFaceAutoLM, self).__init__()  # do the BaseLM init
        self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = model.config

        self._add_special_tokens = add_special_tokens
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = self.max_length

        self.model = model
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        if use_accelerate and "lm_head" in self.model.hf_device_map:
            # `accelerate` can place `lm_head` weights on a different device than
            # the user specified one so we force `self._device` to be the same as
            # `lm_head`'s.
            self._device = self.model.hf_device_map["lm_head"]


def run_lm_eval(model, tokenizer, model_id: str, device: str, task: str, limit=None):
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 4096 if 'llama' in model_id.lower() else 2048
    print(f'Manually setting max_length={max_length} for {model_id} to avoid potential OOM.')

    lm_eval_model = LMEvalModel(model, tokenizer, batch_size=1, max_length=max_length, device=device)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=[task],
        num_fewshot=0,
        batch_size=1,
        no_cache=True,
        limit=limit,
        device=device,
    )
    return results


def get_calibration_texts():
    """
    Sentences from c4 dataset.
    """
    cache_path = './logs/cached_calibration_samples.txt'
    if Path(cache_path).exists():
        with Path(cache_path).open(encoding='utf-8') as f:
            texts = json.load(f)
    else:
        dataset = load_dataset("c4", "en", split="train", streaming=True)
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
        # use llama tokenizer to select long-enough texts
        llama_tokenizer = AutoTokenizer.from_pretrained('yujiepan/llama-2-tiny-random')
        texts = []
        for data in dataset:
            inputs = llama_tokenizer(data['text'], add_special_tokens=False)
            input_ids = inputs['input_ids']
            if not 384 <= len(input_ids) <= 512:
                continue
            texts.append(data['text'])
            if len(texts) >= 512:
                break
        with Path(cache_path).open('w', encoding='utf-8') as f:
            json.dump(texts, f)
    return texts


def create_nncf_dataset(model, tokenizer, device: str, batch_size: int, num_calibration_samples: int):
    all_texts = get_calibration_texts()
    batches = [all_texts[i:i + batch_size] for i in range(0, num_calibration_samples, batch_size)]

    def transform_func(batch):
        inputs = tokenizer(
            batch, truncation=True, return_tensors='pt', max_length=256, padding=False,
        )
        result = {}
        result['input_ids'] = inputs['input_ids'].to(device)
        # result['attention_mask'] = inputs['attention_mask'].to(device)
        return result

    nncf_dataset = nncf.Dataset(batches, transform_func)
    return nncf_dataset


def infer_layer_name(model_id, layer_type: str):
    """
    layer_type: up/gate/down
    """
    model_id = model_id.lower()
    if '/' in model_id:
        author, model_id = model_id.split('/')[-2:]
    else:
        author, model_id = '', model_id
    if 'llama' in model_id or 'mistral' in model_id:
        return f'{{re}}.*{layer_type}_proj.*'
    elif 'mixtral' in model_id:
        if layer_type == 'up':
            return '{re}.*experts.*w3'
        if layer_type == 'gate':
            return '{re}.*experts.*w1'
        if layer_type == 'down':
            return '{re}.*experts.*w2'
    raise NotImplementedError


def get_torch_name(nodename):
    matches = re.findall(r'\[(.*?)\]', nodename)
    return '.'.join(matches)


class SparsifierHook(accelerate.hooks.ModelHook):
    info = defaultdict(list)

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        self.torch_name = get_torch_name(name)

    def post_forward(self, module, output):
        sparsity = (output == 0.).float().mean().cpu().item()
        self.info[self.torch_name].append(sparsity)
        return output
