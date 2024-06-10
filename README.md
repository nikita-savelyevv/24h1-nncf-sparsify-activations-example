# 24h1-nncf-sparsify-activations-example

## Setup

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

## Run example

See `bash.bash`.
