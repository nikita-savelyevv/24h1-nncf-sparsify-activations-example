# 24h1-nncf-sparsify-activations-example

## Setup

Python 3.9+ is required.

```bash
git clone https://github.com/nikita-savelyevv/nncf.git
cd nncf && git checkout activation-sparsity-ov-backend && cd ..
pip install -e nncf
pip install -r requirements.txt

```

## Run
Example command:
```bash
python run_sparsify_activations.py \
--model_id meta-llama/Llama-2-7b-hf \
--torch_dtype float32 --backend ov --device cpu \
--compress_weights_mode int8_asym \
--up 0.32 --gate 0.32 --down 0.52 \
--save_folder ./models/tiny-llama_int8-asym_sparse
```
