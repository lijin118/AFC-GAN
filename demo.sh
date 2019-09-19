#!/usr/bin/env bash

#zsl_cub
CUDA_VISIBLE_DEVICES=0 python3 ./afcgan.py --ensemble_ratio 0.8 --loss_syn_num 5 --cyc_seen_weight 1 --cyc_unseen_weight 1 --dm_seen_weight 0.001 --dm_unseen_weight 0.001 --dm_weight 0.05 --cls_syn_num 1100 --cls_batch_size 1400 --new_lr 0  --nepoch 80  --manualSeed 3483 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --outname cub

#zsl_sun
CUDA_VISIBLE_DEVICES=0 python3 ./afcgan.py --ensemble_ratio 0.9 --loss_syn_num 20 --cyc_seen_weight 1 --cyc_unseen_weight 1 --dm_seen_weight 0.001 --dm_unseen_weight 0.001 --dm_weight 2e-2  --cls_syn_num 450  --cls_batch_size 400  --new_lr 0   --nepoch 60  --manualSeed 4115 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.0005 --outname sun

#zsl_awa
CUDA_VISIBLE_DEVICES=0 python3 ./afcgan.py --ensemble_ratio 0.5 --loss_syn_num 15 --cyc_seen_weight 1 --cyc_unseen_weight 1e-2 --dm_seen_weight 0.001 --dm_unseen_weight     1 --dm_weight 3e-2  --cls_syn_num 850  --cls_batch_size 750  --new_lr 0   --nepoch 40  --manualSeed 9182 --cls_weight 0.01 --preprocessing --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa

#zsl_apy
CUDA_VISIBLE_DEVICES=0 python3 ./afcgan.py --ensemble_ratio 1.1 --loss_syn_num 5 --cyc_seen_weight 10 --cyc_unseen_weight 10 --dm_seen_weight 0.001 --dm_unseen_weight 0.001 --dm_weight 1.5 --cls_syn_num 450  --cls_batch_size 900  --new_lr 0   --nepoch 30  --manualSeed 9182 --cls_weight 0.01 --preprocessing  --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset APY --batch_size 64 --nz 64 --attSize 64 --resSize 2048 --outname apy

# gzsl_cub
CUDA_VISIBLE_DEVICES=0 python3 ./afcgan.py --ensemble_ratio 2.6 --loss_syn_num 5 --cyc_seen_weight 1 --cyc_unseen_weight 1 --dm_seen_weight 0.001 --dm_unseen_weight 0.001 --dm_weight 1e-2  --cls_syn_num 150  --cls_batch_size 150  --new_lr 0  --nepoch 60  --gzsl --manualSeed 3483  --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --nclass_all 200 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --outname cub

# gzsl_sun
CUDA_VISIBLE_DEVICES=0 python3 ./afcgan.py --ensemble_ratio 1.6 --loss_syn_num 20 --cyc_seen_weight 1 --cyc_unseen_weight 1 --dm_seen_weight 0.001 --dm_unseen_weight 0.001 --dm_weight 0.1   --cls_syn_num 40   --cls_batch_size 400  --new_lr 0  --nepoch 50  --gzsl --manualSeed 4115 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.001 --nclass_all 717 --outname sun

# gzsl_awa
CUDA_VISIBLE_DEVICES=0 python3 ./afcgan.py --ensemble_ratio 0.1 --loss_syn_num 30 --cyc_seen_weight 1 --cyc_unseen_weight 1e-4 --dm_seen_weight 0.01 --dm_unseen_weight 1e-4 --dm_weight 1e-3  --cls_syn_num 1900 --cls_batch_size 1650 --new_lr 1  --nepoch 15  --gzsl --manualSeed 9182 --cls_weight 0.01 --preprocessing --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa

# gzsl_apy
CUDA_VISIBLE_DEVICES=0 python3 ./afcgan.py --ensemble_ratio 0.3 --loss_syn_num 10 --cyc_seen_weight 1e-5 --cyc_unseen_weight 1e-4 --dm_seen_weight 0.001 --dm_unseen_weight 0.001 --dm_weight 10    --cls_syn_num 1600 --cls_batch_size 2000 --new_lr 0  --nepoch 40  --gzsl --manualSeed 9182 --cls_weight 0.01 --preprocessing --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 32 --dataset APY --batch_size 64 --nz 64 --attSize 64 --resSize 2048 --outname apy
