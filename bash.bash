# debug
python -u run_sparsify_activations.py --model_id yujiepan/llama-2-tiny-random \
    --torch_dtype float16 --device cuda \
    --compress_weights_mode int8_asym \
    --eval_limit 3 \
    --up 0.3 --gate 0.4 --down 0.5 \
    --save_folder ./models/llama-2-tiny-random/int8_asym_up30+down50/

python -u run_sparsify_activations.py --model_id yujiepan/llama-2-tiny-random \
    --torch_dtype float16 --device cuda \
    --eval_limit 3 \
    --up 0.3 --gate 0.4 --down 0.5 \
    --save_folder ./models/llama-2-tiny-random/up30+down50/

# llama2-7b
python -u run_sparsify_activations.py --model_id meta-llama/Llama-2-7b-hf \
    --torch_dtype float16 --device cuda \
    --compress_weights_mode int8_asym \
    --up 0.3 --gate 0.3 --down 0.5 \
    --save_folder ./models/Llama-2-7b-hf/int8_asym_up30+down50/

python -u run_sparsify_activations.py --model_id meta-llama/Llama-2-7b-hf \
    --torch_dtype float16 --device cuda \
    --up 0.3 --gate 0.3 --down 0.5 \
    --save_folder ./models/Llama-2-7b-hf/up30+down50/

# # mixtral-moe, requires large memory
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u run_sparsify_activations.py --model_id mistralai/Mixtral-8x7B-Instruct-v0.1 \
#     --torch_dtype float16 --device cuda \
#     --compress_weights_mode int8_asym \
#     --up 0.4 --gate 0.4 --down 0.5 \
#     --save_folder ./models/Mixtral-8x7B-Instruct-v0.1/int8_asym_up40+down50/
