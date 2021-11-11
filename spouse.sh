# for ratio in 0 0.5 1.0 # 0 0.25 0.5 0.75 1.0
# do
#     CUDA_VISIBLE_DEVICES=2 python analysis.py --explanation True --task spouse --replace_ratio $ratio --batch_size 2 --gradient_accumulation_steps 16
# done

# for ratio in 0 0.5 1.0 # 0 0.25 0.5 0.75 1.0
# do
#     CUDA_VISIBLE_DEVICES=3 python analysis.py --explanation True --task spouse --replace_ratio $ratio --batch_size 2 --gradient_accumulation_steps 16 --replace_with_new_token True
# done


# for ratio in 0.05 0.1 # 0.2 0.4 0.6 0.8 1.0
# do
#     CUDA_VISIBLE_DEVICES=6 python num_analysis.py --explanation True --task spouse --ratio_train_samples $ratio
# done

# for ratio in 0.4 0.6 0.8
# do
#     CUDA_VISIBLE_DEVICES=7 python num_analysis.py --seed 49 --explanation True --task spouse --ratio_train_samples $ratio --mannual_exp True --batch_size 2 --gradient_accumulation_steps 16
# done

# CUDA_VISIBLE_DEVICES=4 python run_re.py \
    # --explanation True \
    # --mannual_exp True \
    # --task spouse \
    # --batch_size 1 \
    # --gradient_accumulation_steps 32

CUDA_VISIBLE_DEVICES=5 python run_freeze_bert.py \
--task spouse \
--batch_size 32 \
--learning_rate 1e-3 \
--explanation True \
--exp_num 1 \
--num_explanation_tokens 4 \
--epochs 30 \
--dropout 0.0 \
--projection_dim 64 \
--hidden_dim 256 \
--num_layers 0