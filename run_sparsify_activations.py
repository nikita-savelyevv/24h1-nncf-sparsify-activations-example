import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

import transformers
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
from optimum.intel.openvino import OVModelForCausalLM

if True:
    from utils import create_nncf_dataset, infer_layer_name, run_lm_eval


@dataclass
class Args:
    model_id: str = 'yujiepan/llama-2-tiny-random'  # Model should be llama or mistral or mixtral
    torch_dtype: str = 'float16'  # Load the torch model in this dtype
    device: str = field(default='cpu', metadata={'choices': ['cuda', 'cpu']})
    # Whether to do compression before sparsification. Note that torch backend only supports int8.
    compress_weights_mode: str = field(default=None, metadata={'choices': ['int8_sym', 'int8_asym']})

    # Target sparsity for up/gate/down projectors in FFN. Values should be in [0, 1]
    up: float = None
    gate: float = None
    down: float = None

    # Calibration set (from c4 dataset). We use 64 samples in all, each containing 256 tokens.
    batch_size: int = 4
    num_calibration_samples: int = 64

    # evaluate with lm-harness-evaluation
    eval_task: str = field(default='wikitext', metadata={
        'choices': ['wikitext', 'arc_easy', 'arc_challenge', 'boolq', "piqa", 'lambada_openai', 'winogrande', 'sciq', 'hellaswag']
    })
    eval_limit: int = None  # If set, will only evaluate on this many samples. Useful for debugging.
    save_folder: str = './openvino_ir/'


@torch.no_grad()
def main(args: Args):
    Path(args.save_folder).mkdir(exist_ok=True, parents=True)

    # Load model and dataset
    torch_dtype = {'float16': torch.float16, 'float32': torch.float32}[args.torch_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype,
        device_map='auto' if args.device == 'cuda' else 'cpu',
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    nncf_dataset = create_nncf_dataset(
        model, tokenizer, args.device, args.batch_size, args.num_calibration_samples,
    )

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

    # Do activation sparsification
    set_seed(42)
    target_sparsity_by_scope = {}
    if args.up is not None:
        target_sparsity_by_scope[infer_layer_name(args.model_id, 'up')] = args.up
    if args.gate is not None:
        target_sparsity_by_scope[infer_layer_name(args.model_id, 'gate')] = args.gate
    if args.down is not None:
        target_sparsity_by_scope[infer_layer_name(args.model_id, 'down')] = args.down
    print('target_sparsity_by_scope:', target_sparsity_by_scope)
    sparse_model = nncf.experimental.torch.sparsify_activations.sparsify_activations(
        model, nncf_dataset,
        target_sparsity_by_scope=target_sparsity_by_scope,
        ignored_scope=None,
    )
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
        if isinstance(module, nncf.torch.quantization.layers.WeightsDecompressor):
            module.result_dtype = torch.float32
    if args.compress_weights_mode is None:
        sparse_model = sparse_model.float()
    export_from_model(
        sparse_model, args.save_folder, stateful=False, device=args.device,
        compression_option='fp32',
    )
    tokenizer.save_pretrained(args.save_folder)

    # Try loading the IR
    ov_model = OVModelForCausalLM.from_pretrained(args.save_folder)
    pipe = transformers.pipelines.TextGenerationPipeline(model=ov_model, tokenizer=tokenizer)
    output = pipe('Hello, I am an AI chatbot ', max_new_tokens=16)
    print(output)

    # Run wikitext evaluation on OV. This will take long.
    if False:
        ov_eval_results = run_lm_eval(ov_model, tokenizer, args.model_id, 'cpu', args.eval_task, limit=args.eval_limit)
        with open(Path(args.save_folder, 'ov_eval_results.json'), 'w', encoding='utf-8') as f:
            json.dump(ov_eval_results, f, indent=2)
        print('OV evaluation result:', ov_eval_results)


if __name__ == '__main__':
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    print(args)
    main(args)
