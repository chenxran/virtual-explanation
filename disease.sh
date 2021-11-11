# for ratio in 0 0.5 1.0 # 0 0.25 0.5 0.75 1.0
# do
    # CUDA_VISIBLE_DEVICES=0 python analysis.py --explanation True --task disease --replace_ratio $ratio --batch_size 4 --gradient_accumulation_steps 8
# done

# for ratio in 0 0.5 1.0 # 0 0.25 0.5 0.75 1.0
# do
#     CUDA_VISIBLE_DEVICES=1 python analysis.py --explanation True --task disease --replace_ratio $ratio --batch_size 4 --gradient_accumulation_steps 8 --replace_with_new_token True
# done


# for ratio in 0.05 0.1 # 0.2 0.4 0.6 0.8 1.0
# do
#     CUDA_VISIBLE_DEVICES=4 python num_analysis.py --explanation True --task disease --ratio_train_samples $ratio
# done

# for ratio in 0.4 0.8 1.0
# do
#     CUDA_VISIBLE_DEVICES=0 python num_analysis.py --seed 45 --explanation True --task disease --ratio_train_samples $ratio --mannual_exp True --batch_size 2 --gradient_accumulation_steps 16
# done

# CUDA_VISIBLE_DEVICES=0 python run_re.py \
#     --explanation True \
#     --task disease \
    # --batch_size 32

# CUDA_VISIBLE_DEVICES=0 python analysis_new.py --trainset disease-train --testset disease-test --task disease --replace_ratio 0 --batch_size 32

# CUDA_VISIBLE_DEVICES=0 python analysis_new.py --trainset disease-train --testset disease-test --task disease --replace_ratio 0 --batch_size 32 --replacee_with_new_token True

# CUDA_VISIBLE_DEVICES=0 python analysis_new.py --trainset spouse-train --testset spouse-test --task spouse --replace_ratio 0 --batch_size 32

# CUDA_VISIBLE_DEVICES=0 python analysis_new.py --trainset spouse-train --testset spouse-test --task spouse --replace_ratio 0 --batch_size 32 --replace_with_new_token True

CUDA_VISIBLE_DEVICES=6 python run_freeze_bert.py \
--task disease \
--batch_size 32 \
--learning_rate 1e-3 \
--explanation True \
--exp_num 1 \
--num_explanation_tokens 4 \
--epochs 30 \
--dropout 0.0 \
--projection_dim 768 \
--hidden_dim 256 \
--num_layers 0