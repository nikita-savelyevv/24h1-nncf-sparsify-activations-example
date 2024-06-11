from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import transformers

import openvino as ov

import nncf
import nncf.experimental
import nncf.experimental.torch.sparsify_activations
from nncf.scopes import IgnoredScope


class ThreeLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(8, 32, bias=False)
        self.linear3 = nn.Linear(16, 8)

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        y0 = self.linear3(self.linear1(x))
        y1 = self.linear2(x)
        return y0, y1


def dummy_llama_model():
    config = transformers.models.llama.configuration_llama.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        use_cache=False,
        return_dict=False,
    )
    model = transformers.AutoModelForCausalLM.from_config(config)
    return model


@dataclass
class ModelDesc:
    name: str
    model_getter: Callable[[], nn.Module]
    dataset_getter: Callable[[torch.device], nncf.Dataset]
    target_sparsity_by_scope: Dict[str, float]
    ignored_scope: Optional[nncf.IgnoredScope]


model_list = [
    ModelDesc(
        name="linear",
        model_getter=lambda: nn.Linear(8, 16),
        dataset_getter=lambda: nncf.Dataset(torch.randn([3, 2, 8])),
        target_sparsity_by_scope={
            "{re}.*linear.*": 0.3,
        },
        ignored_scope=None,
    ),
    ModelDesc(
        name="three_linear",
        model_getter=ThreeLinearModel,
        dataset_getter=lambda: nncf.Dataset(torch.randint(0, 30, (3, 2, 8))),
        target_sparsity_by_scope={
            "{re}.*linear.*": 0.4,
        },
        ignored_scope=None,
    ),
    ModelDesc(
        name="dummy_llama",
        model_getter=dummy_llama_model,
        dataset_getter=lambda: nncf.Dataset(torch.randint(0, 30, (3, 2, 8))),
        target_sparsity_by_scope={
            "{re}.*gate_proj.*": 0.2,
            "{re}.*up_proj.*": 0.3,
            "{re}.*down_proj.*": 0.4,
        },
        ignored_scope=None,
    ),
]


def export_sparse_ir(desc: ModelDesc, compress_weights: bool):
    print(desc, 'compress_weights=', compress_weights)
    model = desc.model_getter().eval()
    dataset = desc.dataset_getter()
    if compress_weights:
        model = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT8_ASYM,
            dataset=dataset,
        )
    model = nncf.experimental.torch.sparsify_activations.sparsify_activations(
        model=model,
        dataset=dataset,
        target_sparsity_by_scope=desc.target_sparsity_by_scope,
        ignored_scope=desc.ignored_scope,
    )
    example_input = next(iter(dataset.get_inference_data()))
    ov_model = ov.convert_model(model, example_input=example_input)
    compiled_model = ov.compile_model(ov_model, "CPU", config={ov.properties.hint.inference_precision: "f32"})
    compiled_model(example_input.cpu())

    model_path = f'./dummy_models/{desc.name}_int8_sparse.xml' if compress_weights else f'./dummy_models/{desc.name}_sparse.xml'
    ov.save_model(ov_model, model_path, compress_to_fp16=False)


if __name__ == '__main__':
    for compress_weights in [False, True]:
        for desc in model_list:
            export_sparse_ir(desc, compress_weights)
