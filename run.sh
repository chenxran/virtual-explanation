for task in disease spouse
do
    for ratio in 0 0.25 0.5 0.75 1.0
    do
        CUDA_VISIBLE_DEVICES=0 python analysis.py --explanation True --task $task --mannual_exp True --replace_ratio $ratio --gradient_accumulation_steps 32 --batch_size 1
    done
done

# for task in spouse disease
# do
#     for ratio in 0 0.25 0.5 0.75 1.0
#     do
#         CUDA_VISIBLE_DEVICES=1 python analysis.py --explanation True --task $task --mannual_exp True --replace_ratio $ratio --gradient_accumulation_steps 32 --batch_size 1 --replace_with_new_token True
#     done
# done