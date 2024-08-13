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
from nncf.experimental.torch.sparsify_activations import TargetScope


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


class MatMulModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(8, 16)

    def forward(self, x: torch.Tensor):
        y1 = self.linear1(x)
        y2 = self.linear2(x)
        y3 = torch.matmul(y1, y2.T)
        return y3


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
    target_sparsity_by_scope: Dict[TargetScope, float]
    ignored_scope: Optional[nncf.IgnoredScope]


model_list = [
    ModelDesc(
        name="linear",
        model_getter=lambda: nn.Linear(8, 16),
        dataset_getter=lambda: nncf.Dataset(torch.randn([3, 2, 8])),
        target_sparsity_by_scope={
            TargetScope(patterns=[".*linear.*"]): 0.3,
        },
        ignored_scope=None,
    ),
    ModelDesc(
        name="matmul",
        model_getter=MatMulModel,
        dataset_getter=lambda: nncf.Dataset(torch.randn([3, 2, 8])),
        target_sparsity_by_scope={
            TargetScope(patterns=[".*linear.*"]): 0.3,
        },
        ignored_scope=None,
    ),
    ModelDesc(
        name="three_linear",
        model_getter=ThreeLinearModel,
        dataset_getter=lambda: nncf.Dataset(torch.randint(0, 30, (3, 2, 8))),
        target_sparsity_by_scope={
            TargetScope(patterns=[".*linear.*"]): 0.4,
        },
        ignored_scope=None,
    ),
    ModelDesc(
        name="dummy_llama",
        model_getter=dummy_llama_model,
        dataset_getter=lambda: nncf.Dataset(torch.randint(0, 30, (3, 2, 8))),
        target_sparsity_by_scope={
            TargetScope(patterns=[".*gate_proj.*"]): 0.2,
            TargetScope(patterns=[".*up_proj.*"]): 0.3,
            TargetScope(patterns=[".*down_proj.*"]): 0.4,
        },
        ignored_scope=None,
    ),
]


def export_sparse_ir(desc: ModelDesc, ov_backend: bool, compression_bits: Optional[int]):
    assert compression_bits in [None, 4, 8]
    assert ov_backend or compression_bits != 4
    model = desc.model_getter().eval()
    dataset = desc.dataset_getter()
    example_input = next(iter(dataset.get_inference_data()))
    if ov_backend:
        model = ov.convert_model(model, example_input=example_input)
    if compression_bits is not None:
        kwargs = {
            "mode": nncf.CompressWeightsMode.INT8_ASYM if compression_bits == 8 else nncf.CompressWeightsMode.INT4_ASYM,
            "ignored_scope": desc.ignored_scope
        }
        if not ov_backend:
            kwargs["dataset"] = dataset
        if compression_bits == 4:
            kwargs["group_size"] = 1
        model = nncf.compress_weights(model, **kwargs)
    model = nncf.experimental.torch.sparsify_activations.sparsify_activations(
        model=model,
        dataset=dataset,
        target_sparsity_by_scope=desc.target_sparsity_by_scope,
        ignored_scope=desc.ignored_scope,
    )
    if not ov_backend:
        model = ov.convert_model(model, example_input=example_input)
    compiled_model = ov.compile_model(model, "CPU", config={ov.properties.hint.inference_precision: "f32"})
    compiled_model(example_input.cpu())

    save_dir = Path(f"./dummy_models/{'ov' if ov_backend else 'pt'}")
    save_filename = f"{desc.name}{'_int8' if compression_bits == 8 else '_int4' if compression_bits == 4 else ''}_sparse.xml"
    ov.save_model(model, save_dir / save_filename, compress_to_fp16=False)


if __name__ == '__main__':
    for desc in model_list:
        for ov_backend in [True, False]:
            for compression_bits in [None, 4, 8]:
                if not ov_backend and compression_bits == 4:
                    continue
                export_sparse_ir(desc, ov_backend, compression_bits)
