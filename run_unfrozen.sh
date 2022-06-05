

# '--run should be one of the following:'
# 'baseline, baseline-without-place-holder, virtual-explanation,'
# 'virtual-explanation-custom-exp-num, annotated-explanation, mixture,'
# 'fixed-random, factorized-random, factorized-dense, dense.'

GPU_ID=6
RUN=virtual-explanation
MODEL=roberta-base
TASK=spouse

CUDA_VISIBLE_DEVICES=$GPU_ID python run_re.py \
    --model $MODEL \
    --task $TASK \
    --place_holder \
    --epoch 10 \
    --run $RUN
