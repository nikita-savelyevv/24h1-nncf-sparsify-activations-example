# 24h1-nncf-sparsify-activations-example

## Setup

**Note: I tested on torch==2.1.0. Some higher version (e.g. 2.3.1) might have issues with ov export.**

```bash
# install nncf
git clone https://github.com/yujiepan-work/nncf.git
cd nncf
git checkout 24h1/sparse-activation/nncf-pr
pip install -e .

# tested package versions
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@906ef948dc8dbb4c84e1bb0f2861b1aba30ab533
pip install transformers==4.39.3
pip install optimum-intel==1.17.2
```

## Dummy models

`python run_small_model.py`

See the example IR models at `dummy_models` folder.

## Run example to export the model

See `bash.bash`. The exported models should be at `./models`

Reference llama2-7b sparse IR:

- INT8_ASYM + Sparse: <https://huggingface.co/yujiepan/Llama-2-7b-int8asym-sparse-up30-gate30-down50>
- FP32 + Sparse: <https://huggingface.co/yujiepan/Llama-2-7b-sparse-up30-gate30-down50>
- FP16 + Sparse (converted from "FP32+Sparse" by `ov_model.half()` for smaller size): <https://huggingface.co/yujiepan/Llama-2-7b-fp16-sparse-up30-gate30-down50>

## Load the OV model

```python
import transformers
from optimum.intel.openvino import OVModelForCausalLM
model_id = '<local folder or model_id on HF>'
ov_model = OVModelForCausalLM.from_pretrained(model_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
pipe = transformers.pipelines.TextGenerationPipeline(model=ov_model, tokenizer=tokenizer)
output = pipe('Hello, I am a ', max_new_tokens=16)
```
