export PYTHONPATH="/root/workspace/arithmetic/AdaMerging/"
CUDA_VISIBLE_DEVICES=1 python finetune.py --preweight "openai"
# CUDA_VISIBLE_DEVICES=0 python finetune.py --preweight "laion2b_e16"
# CUDA_VISIBLE_DEVICES=0 python finetune.py --preweight "laion400m_e31"
# CUDA_VISIBLE_DEVICES=0 python finetune.py --preweight "laion400m_e32"
# CUDA_VISIBLE_DEVICES=3 python finetune.py --preweight "datacomp_xl_s13b_b90k"
# CUDA_VISIBLE_DEVICES=3 python finetune.py --preweight "datacomp_m_s128m_b4k"
# CUDA_VISIBLE_DEVICES=3 python finetune.py --preweight "datacomp_s_s13m_b4k"

