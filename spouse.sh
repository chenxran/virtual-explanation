# for ratio in 0.25 0.75  # 0 0.5 1.0 # 0 0.25 0.5 0.75 1.0
# do
#     CUDA_VISIBLE_DEVICES=5 python analysis.py --explanation True --task spouse --replace_ratio $ratio --batch_size 2 --gradient_accumulation_steps 16
# done

# for ratio in 0 0.5 1.0 # 0 0.25 0.5 0.75 1.0
# do
    # CUDA_VISIBLE_DEVICES=3 python analysis.py --explanation True --task spouse --replace_ratio $ratio --batch_size 2 --gradient_accumulation_steps 16 --replace_with_new_token True
# done


# for ratio in 0.05 0.1 0.2 0.4  # 0.6 0.8 1.0
# do
#     CUDA_VISIBLE_DEVICES=1 python num_analysis.py --task spouse --ratio_train_samples $ratio --explanation True --epochs 20
# done

# for ratio in 0.05 0.1 0.2 0.4
# do
#     CUDA_VISIBLE_DEVICES=7 python num_analysis.py --explanation True --task spouse --ratio_train_samples $ratio --mannual_exp True --batch_size 2 --gradient_accumulation_steps 16
# done

# CUDA_VISIBLE_DEVICES=4 python run_re.py \
#     --explanation True \
#     --task spouse \
#     --batch_size 32 \
#     --no_place_holder True \
#     --epochs 30
    # --gradient_accumulation_steps 32

# CUDA_VISIBLE_DEVICES=5 python run_freeze_bert.py \
# --task spouse \
# --batch_size 32 \
# --learning_rate 1e-3 \
# --explanation True \
# --exp_num 1 \
# --num_explanation_tokens 4 \
# --epochs 30 \
# --dropout 0.0 \
# --projection_dim 64 \
# --hidden_dim 256 \
# # --num_layers 0
# for ratio in 0.75
# do
#     CUDA_VISIBLE_DEVICES=7 python freeze_analysis.py --task spouse --batch_size 2 --learning_rate 1e-3 --explanation True --replace_ratio $ratio --epochs 30 --dropout 0.0 --projection_dim 64 --hidden_dim 256 --num_layers 0 --gradient_accumulation_steps 16
# done
# CUDA_VISIBLE_DEVICES=7 python freeze_analysis.py --task spouse --batch_size 2 --learning_rate 1e-3 --explanation True --replace_ratio 0.75 --epochs 30 --dropout 0.0 --projection_dim 64 --hidden_dim 256 --num_layers 0 --gradient_accumulation_steps 16 --replace_with_new_token True


# chong pao
# liang bian meiyou
# liang bian meiyou jiewei mei juhao
# long num explanation tokens
# short
CUDA_VISIBLE_DEVICES=0 python run_re.py \
--task spouse \
--explanation True \
--batch_size 16 \
--gradient_accumulation_steps 2 \
--exp_num 3
# --num_explanation_tokens 2


# for ratio in 0.05 0.1 #  0.2 0.4
# do
#     CUDA_VISIBLE_DEVICES=4 python num_analysis.py --explanation True --task spouse --ratio_train_samples $ratio --mannual_exp True --batch_size 2 --gradient_accumulation_steps 16 --epochs 20
# done


# for ratio in 0.1 # 0.2 0.4
# do
#     CUDA_VISIBLE_DEVICES=5 python random_num_analysis.py --explanation True --task spouse --ratio_train_samples $ratio --batch_size 2 --gradient_accumulation_steps 16 --replace_ratio 1.0 --epochs 20
# done