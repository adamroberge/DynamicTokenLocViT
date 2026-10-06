# Historical command notebook. Run individual entries from the repository root.
# Launch paths use the current modules; experiment settings and local paths are preserved.

# ImageNet1k adamw
python -m torch.distributed.launch --nproc_per_node=4 --use_env --module training.main --model vit_register_dynamic_viz --data-path /home/adam/data/in1k --data-set IMNET --batch-size 256 --lr 4e-3 --epochs 300 --weight-decay 0.05 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment

# ImageNet1k lamb
python -m torch.distributed.launch --nproc_per_node=4 --use_env --module training.main --model vit_register_dynamic_viz --data-path /home/adam/data/in1k --data-set IMNET --device cuda --batch-size 1024 --lr 4e-3 --epochs 300 --weight-decay 0.02 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt lamb --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment

# ImageNet1k simple runs
torchrun --nnodes=1 --nproc_per_node=4 --module training.main --model vit_register_dynamic_viz --data-path /home/adam/data/in1k --data-set IMNET --device cuda --batch-size 256 --lr 4e-3 --epochs 400 --weight-decay 0.02 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt lamb --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment
torchrun --nnodes=1 --nproc_per_node=4 --module training.main
torchrun --nnodes=1 --nproc_per_node=4 --module training.main --distributed

# ImageNet1k - Setting the token locations dynamically
torchrun --nnodes=1 --nproc_per_node=3 --module training.main --distributed --num_reg 4 --cls_pos 6 --reg_pos 
torchrun --nnodes=1 --nproc_per_node=3 --module training.main --distributed --lr 5e-6 --num_reg 4 --cls_pos 6 --reg_pos 3
python -m torch.distributed.launch  --nproc_per_node=3 --use_env --module training.main --distributed --model vit_register_dynamic_viz --data-path /home/adam/data/in1k --data-set IMNET --device cuda --batch-size 1024 --lr 5e-6 --epochs 300 --weight-decay 0.02 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt lamb --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment --num_reg 4 --cls_pos 6 --reg_pos 3


# ImageNet1k + distillation - Setting the token locations dynamically
torchrun --nnodes=1 --nproc_per_node=4 --module training.distillation --distributed --num_reg 4 --cls_pos 6 --reg_pos 3


# CIFAR10 
python -m training.main --model vit_register_dynamic_viz --data-path /data/CIFAR10/cifar-10-batches-py --data-set CIFAR10 --device cuda --batch-size 256 --lr 4e-3 --epochs 300 --weight-decay 0.05 --sched cosine --input-size 32 --patch-size 4 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 10 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment

# ImageNet1k - training/imagenet/main.py (simple)
python -m training.imagenet.main --data-path /home/adam/data/in1k --device cuda --gpu "0,1,2" --batch-size 512 --lr 4e-3 --epochs 100 --weight-decay 0.05 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment


# Visualization 

# match the cls_pos and reg_pos with the trained model - best_checkpoint_cls_{cls_pos}_reg_{reg_pos}
# layer_num should be larger than cls_pos and reg_pos
# choose any image (img_num < batch_size)

# ImageNet1k - training/imagenet/main.py (simple)
python -m visualization.imagenet --cls_pos 0 --reg_pos 0 --layer_num 3 --img_num 0 
python -m visualization.imagenet --model_path /home/adam/dynamic_vit/DynamicTokenLocViT/result/best_checkpoint.pth --cls_pos 0 --reg_pos 0 --layer_num 3 --img_num 0 

# ImageNet1k - training/main.py 
# make sure to change model path to match the cls_pos and reg_pos 
python -m visualization.attention --model_path /home/adam/dynamic_vit/DynamicTokenLocViT/result/best_checkpoint_cls_0_reg_0.pth --cls_pos 0 --reg_pos 0 --layer_num 3 --img_num 0 
python -m visualization.attention --model_path /home/adam/dynamic_vit/DynamicTokenLocViT/result/best_checkpoint.pth --cls_pos 0 --reg_pos 0 --layer_num 1 --img_num 0
