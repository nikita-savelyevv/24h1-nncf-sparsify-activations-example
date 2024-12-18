import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import openvino as ov
import torch
import transformers
from openvino._pyopenvino._offline_transformations import compress_model_transformation
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

import nncf
import nncf.experimental
import nncf.experimental.torch.sparsify_activations
import nncf.experimental.torch.sparsify_activations.torch_backend
import nncf.quantization
import nncf.torch
import nncf.torch.quantization
import nncf.torch.quantization.quantize_model
from nncf import CompressWeightsMode
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig, OVConfig

from nncf.experimental.torch.sparsify_activations import TargetScope
from utils import create_nncf_dataset_pt, create_nncf_dataset_ov, infer_layer_name, run_lm_eval, get_ov_input_shapes


@dataclass
class Args:
    # Model should be llama or mistral or mixtral
    model_id: str = 'yujiepan/llama-2-tiny-random'
    torch_dtype: str = 'float16'  # Load the torch model in this dtype
    device: str = field(default='cpu', metadata={'choices': ['cuda', 'cpu']})
    # Whether to do compression before sparsification. Note that torch backend only supports int8.
    compress_weights_mode: str = field(default=None, metadata={'choices': [
        'int8_sym',
        'int8_asym',
        'int4_sym',
        'int4_asym',
        'int4',
    ]})

    ratio: float = None  # If set, will compress the model to this ratio.

    # Target sparsity for up/gate/down projectors in FFN. Values should be in [0, 1]
    up: float = None
    gate: float = None
    down: float = None

    # Calibration set (from c4 dataset). We use 64 samples in all, each containing 256 tokens.
    batch_size: int = 1
    num_calibration_samples: int = 64

    # evaluate with lm-harness-evaluation
    eval_task: str = field(default='wikitext', metadata={
        'choices': ['wikitext', 'arc_easy', 'arc_challenge', 'boolq', "piqa",
                    'lambada_openai', 'winogrande', 'sciq', 'hellaswag']
    })
    eval_limit: int = None  # If set, will only evaluate on this many samples. Useful for debugging.
    save_folder: str = './models'
    backend: str = field(default=None, metadata={'choices': ['pt', 'ov']})


@torch.no_grad()
def main(args: Args):
    save_path = Path(args.save_folder)
    save_path.mkdir(exist_ok=True, parents=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if args.backend == 'pt':
        nncf_dataset = create_nncf_dataset_pt(
            tokenizer, args.device, args.batch_size, args.num_calibration_samples
        )

        # Load model and dataset
        torch_dtype = {'float16': torch.float16, 'float32': torch.float32}[args.torch_dtype]
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch_dtype,
            device_map='auto' if args.device == 'cuda' else 'cpu',
            # HF uses sdpa by default since torch>=2.1.1, which is not well supported by OV export.
            attn_implementation="eager",
        )
        model = model.eval()

        # Do weight compression if needed
        set_seed(42)
        if args.compress_weights_mode is not None:
            compression_mode = {
                'int8_sym': CompressWeightsMode.INT8_SYM,
                'int8_asym': CompressWeightsMode.INT8_ASYM,
            }.get(args.compress_weights_mode)
            model = nncf.compress_weights(
                model,
                mode=compression_mode,
                dataset=nncf_dataset,
            )
    else:
        quantization_config = None
        if args.compress_weights_mode is not None:
            quantization_config = {}
            if "int4" in args.compress_weights_mode:
                quantization_config["bits"] = 4
                if "sym" in args.compress_weights_mode:
                    quantization_config["group_size"] = 128
            else:
                quantization_config["bits"] = 8
            if "asym" in args.compress_weights_mode:
                quantization_config["sym"] = False
            elif "sym" in args.compress_weights_mode:
                quantization_config["sym"] = True
            if args.ratio is not None:
                quantization_config["ratio"] = args.ratio
        model_fn = lambda: OVModelForCausalLM.from_pretrained(
            args.model_id,
            export=True,
            # HF uses sdpa by default since torch>=2.1.1, which is not well supported by OV export.
            attn_implementation="eager",
            quantization_config=quantization_config,
            compile=False
        )
        if args.compress_weights_mode is None:
            with patch('optimum.exporters.openvino.convert._MAX_UNCOMPRESSED_SIZE', float('inf')):
                model = model_fn()
            compress_model_transformation(model.model)
        else:
            model = model_fn()
        model.save_pretrained(save_path)
        model = model.model

        batch_size = 1
        nncf_dataset = create_nncf_dataset_ov(
            tokenizer, batch_size, num_calibration_samples=args.num_calibration_samples,
            input_shapes=get_ov_input_shapes(model, batch_size)
        )

    # Do activation sparsification
    set_seed(42)
    target_sparsity_by_scope = {}
    if args.up is not None:
        target_sparsity_by_scope[TargetScope(patterns=[infer_layer_name(args.model_id, 'up')])] = args.up
    if args.gate is not None:
        target_sparsity_by_scope[TargetScope(patterns=[infer_layer_name(args.model_id, 'gate')])] = args.gate
    if args.down is not None:
        target_sparsity_by_scope[TargetScope(patterns=[infer_layer_name(args.model_id, 'down')])] = args.down
    if args.up or args.gate or args.down:
        print('target_sparsity_by_scope:', target_sparsity_by_scope)
        sparse_model = nncf.experimental.torch.sparsify_activations.sparsify_activations(
            model, nncf_dataset,
            target_sparsity_by_scope=target_sparsity_by_scope,
            ignored_scope=None,
        )
    else:
        sparse_model = model
        print("No sparsification target sparsity is set. Skipping sparsification.")

    if args.backend == 'pt':
        print(sparse_model)

        # Run wikitext evaluation
        torch_eval_results = run_lm_eval(
            model, tokenizer, args.model_id, args.device,
            task=args.eval_task, limit=args.eval_limit
        )
        with open(Path(args.save_folder, 'torch_eval_results.json'), 'w', encoding='utf-8') as f:
            json.dump(torch_eval_results, f, indent=2)
        print('Torch evaluation result:', torch_eval_results)

        # Openvino IR export.
        # This is a bit tricky here. We ensure the inference precision is in FP32 otherwise we will get errors
        # when exporting to IR.
        for module in sparse_model.nncf.modules():
            if isinstance(module, nncf.torch.quantization.layers.AsymmetricWeightsDecompressor) or \
               isinstance(module, nncf.torch.quantization.layers.SymmetricWeightsDecompressor):
                module.result_dtype = torch.float32
        if args.compress_weights_mode is None:
            sparse_model = sparse_model.float()
        # Optimum-intel will do weight compression without asking user if the model is
        # larger than _MAX_UNCOMPRESSED_SIZE. Disable it so that we can export a float model.
        with patch('optimum.exporters.openvino.convert._MAX_UNCOMPRESSED_SIZE', float('inf')):
            export_from_model(
                sparse_model, save_path, stateful=False, device=args.device,
                compression_option='fp32',
            )
    else:
        ov.save_model(sparse_model, save_path / "openvino_model.xml")
    tokenizer.save_pretrained(save_path)

    # Try loading the IR
    ov_model = OVModelForCausalLM.from_pretrained(save_path)
    pipe = transformers.pipelines.TextGenerationPipeline(model=ov_model, tokenizer=tokenizer)
    output = pipe('Hello, I am an AI chatbot ', max_new_tokens=16)
    print(output)

    # Run wikitext evaluation on OV. This will take long.
    ov_eval_results = run_lm_eval(
        ov_model, tokenizer, args.model_id, 'cpu', args.eval_task, limit=args.eval_limit
    )
    with open(Path(save_path, 'ov_eval_results.json'), 'w', encoding='utf-8') as f:
        json.dump(ov_eval_results, f, indent=2)
    print('OV evaluation result:', ov_eval_results)


if __name__ == '__main__':
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    print(args)
    main(args)
