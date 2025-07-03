#!/bin/bash 
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=30


echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
CUDA_DEVICE=$(echo "$CUDA_VISIBLE_DEVICES," | cut -d',' -f $((SLURM_LOCALID + 1)) );

T_REGEX='^[0-9]$';
if ! [[ "$CUDA_DEVICE" =~ $T_REGEX ]]; then
        echo "error no reserved gpu provided"
        exit 1;
fi
echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID using gpu id              +++$CUDA_DEVICE (we may use gpu: $CUDA_VISIBLE_DEVICES on $(hostname))"
echo "computing on $(nvidia-smi --query-gpu=gpu_name --format=csv -i $CUDA_DEVICE | tail -n 1)"

OMP_NUM_THREADS=1
torchrun --nproc_per_node=2 run_mae_pretrain_FedAvg.py  --data_path /homes/xwang/mtan/NewExp_4clients_cls36/SSLpretrain --data_set SSLpretrain_FL  --output_dir /homes/xwang/mtan/4cluster/SSL-FL/data/ckpts/NewExp_4clients_cls36/fed_mae/pretrained_split_01_800ep_blr1.5e-4_bs128_dis2_seed0_HVnoP_scale0.7   --split_type split_01  --save_ckpt_freq 800    --model mae_vit_base_patch16   --batch_size 128  --blr 1.5e-4  --mask_ratio 0.75 --weight_decay 0.05  --warmup_epochs 40  --norm_pix_loss --sync_bn --n_clients 4 --E_epoch 1 --max_communication_rounds 800 --num_local_clients -1
echo "done"

