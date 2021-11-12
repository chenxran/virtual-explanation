# for ratio in 0 0.5 1.0
# do
#     CUDA_VISIBLE_DEVICES=0 python analysis.py --explanation True --task spouse --replace_ratio $ratio --batch_size 1 --gradient_accumulation_steps 32
# done

# for ratio in 0 0.5 1.0
# do
    # CUDA_VISIBLE_DEVICES=1 python analysis.py --explanation True --task spouse --replace_ratio $ratio --batch_size 1 --gradient_accumulation_steps 32 --replace_with_new_token True
# done


CUDA_VISIBLE_DEVICES=7 python analysis.py --explanation True --task tacred --replace_ratio 0 --batch_size 1 --gradient_accumulation_steps 32
# CUDA_VISIBLE_DEVICES=1 python analysis.py --explanation True --task tacred --replace_ratio 0.5 --batch_size 1 --gradient_accumulation_steps 32
# CUDA_VISIBLE_DEVICES=2 python analysis.py --explanation True --task tacred --replace_ratio 1.0 --batch_size 1 --gradient_accumulation_steps 32
# CUDA_VISIBLE_DEVICES=3 python analysis.py --explanation True --task tacred --replace_ratio 0.5 --batch_size 1 --gradient_accumulation_steps 32 --replace_with_new_token True
# CUDA_VISIBLE_DEVICES=4 python analysis.py --explanation True --task tacred --replace_ratio 1.0 --batch_size 1 --gradient_accumulation_steps 32 --replace_with_new_token True

# for ratio in 0.05 0.1 0.2 0.4
# do
#     CUDA_VISIBLE_DEVICES=3 python num_analysis.py --explanation True --task spouse --ratio_train_samples $ratio
# done

# CUDA_VISIBLE_DEVICES=3 python num_analysis.py --explanation True --task tacred --ratio_train_samples 0.05
# CUDA_VISIBLE_DEVICES=4 python num_analysis.py --explanation True --task tacred --ratio_train_samples 0.1
# CUDA_VISIBLE_DEVICES=5 python num_analysis.py --explanation True --task tacred --ratio_train_samples 0.2 --batch_size 2 --gradient_accumulation_steps 16
# CUDA_VISIBLE_DEVICES=6 python num_analysis.py --explanation True --task tacred --ratio_train_samples 0.4

# CUDA_VISIBLE_DEVICES=3 python num_analysis.py --explanation True --task tacred --ratio_train_samples 0.05 --mannual_exp True --batch_size 1 --gradient_accumulation_steps 32
# CUDA_VISIBLE_DEVICES=4 python num_analysis.py --explanation True --task tacred --ratio_train_samples 0.1 --mannual_exp True --batch_size 1 --gradient_accumulation_steps 32
# CUDA_VISIBLE_DEVICES=5 python num_analysis.py --explanation True --task tacred --ratio_train_samples 0.2 --mannual_exp True --batch_size 1 --gradient_accumulation_steps 32
# CUDA_VISIBLE_DEVICES=6 python num_analysis.py --explanation True --task tacred --ratio_train_samples 0.4 --mannual_exp True --batch_size 1 --gradient_accumulation_steps 32

# CUDA_VISIBLE_DEVICES=3 python num_analysis.py --task tacred --ratio_train_samples 0.05
# CUDA_VISIBLE_DEVICES=3 python num_analysis.py --task tacred --ratio_train_samples 0.1
# CUDA_VISIBLE_DEVICES=7 python num_analysis.py --task tacred --ratio_train_samples 0.2
# CUDA_VISIBLE_DEVICES=3 python num_analysis.py --task tacred --ratio_train_samples 0.4