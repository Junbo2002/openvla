torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir /home/ma-user/obs/wangjunbo/datasets/bridge_dataset/datasets \
  --run_root_dir /home/ma-user/obs/wangjunbo/code/openvla/openvla/logs \
  --wandb_entity "junbowang"

export OMP_NUM_THREADS=24
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir /home/ma-user/obs/wangjunbo/datasets/bridge_dataset/datasets \
  --run_root_dir /home/ma-user/obs/wangjunbo/code/openvla/openvla/logs \
  --wandb_entity "junbowang" 

python --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir /home/ma-user/obs/wangjunbo/datasets/bridge_dataset/datasets \
  --run_root_dir /home/ma-user/obs/wangjunbo/code/openvla/openvla/logs \
  --wandb_entity "junbowang" 

# fine-tuning from pre-trained model
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint /home/ma-user/obs/wangjunbo/models/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir /home/ma-user/obs/wangjunbo/datasets/bridge_dataset/datasets \
  --run_root_dir /home/ma-user/obs/wangjunbo/code/openvla/openvla/logs \
  --image_aug False \
  --wandb_project "openvla-pretrained" \
  --wandb_entity "junbowang" \
  --save_interval 2500 \
  --is_resume False