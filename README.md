# 24h1-nncf-sparsify-activations-example

## Setup

Python 3.9+ is required.

```bash
pip install git+https://github.com/nikita-savelyevv/nncf.git@activation-sparsity-ov-backend
pip install -r requirements.txt

```

## Run
Example command:
```bash
python run_sparsify_activations.py \
--model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
--torch_dtype float32 --backend ov --device cpu \
--compress_weights_mode int8_asym \
--up 0.32 --gate 0.32 --down 0.52 \
--save_folder ./models/tiny-llama_int8-asym_sparse
```
