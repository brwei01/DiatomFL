#!/bin/bash 
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=30


CUDA_DEVICE=$(echo "$CUDA_VISIBLE_DEVICES," | cut -d',' -f $((SLURM_LOCALID + 1)) );

T_REGEX='^[0-9]$';
if ! [[ "$CUDA_DEVICE" =~ $T_REGEX ]]; then
        echo "error no reserved gpu provided"
        exit 1;
fi
echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID using gpu id              +++$CUDA_DEVICE (we may use gpu: $CUDA_VISIBLE_DEVICES on $(hostname))"
echo "computing on $(nvidia-smi --query-gpu=gpu_name --format=csv -i $CUDA_DEVICE | tail -n 1)"

OMP_NUM_THREADS=1
torchrun --nproc_per_node=2 run_class_finetune_FedAvg.py  --data_path /homes/xwang/mtan/NewExp_4clients_cls36/SSLfinetune --data_set datasets_diatom_FSSL   --finetune /homes/xwang/mtan/4cluster/SSL-FL/data/ckpts/NewExp_4clients_cls36/fed_mae/pretrained_split_iid_800ep_blr1.5e-4_bs128_dis2_seed0_HVnoP_scale0.7-1/checkpoint-799.pth    --nb_classes 36     --output_dir /homes/xwang/mtan/4cluster/SSL-FL/data/ckpts/NewExp_4clients_cls36/fed_mae/pretrained_split_iid_800ep_blr1.5e-4_bs128_dis2_seed0_HVnoP_scale0.7-1/split_dbar1.7_std1.0_blr1e-3_seed0_0  --split_type split_dbar1.7_std1.0   --save_ckpt_freq 100    --model vit_base_patch16     --batch_size 64   --blr 1e-3 --layer_decay 0.75     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0 --cutmix 0    --n_clients 4 --E_epoch 1 --max_communication_rounds 100 --num_local_clients -1
echo "done"

