# ImageNet1k adamw
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model vit_register_dynamic_viz --data-path /home/adam/data/in1k --data-set IMNET --batch-size 256 --lr 4e-3 --epochs 300 --weight-decay 0.05 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment

# ImageNet1k lamb
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 main.py --model vit_register_dynamic_viz --data-path /home/adam/data/in1k --data-set IMNET --device cuda --batch-size 1024 --lr 4e-3 --epochs 400 --weight-decay 0.02 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt lamb --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment

# ImageNet1k simple runs
torchrun --nnodes=1 --nproc_per_node=4 main.py --model vit_register_dynamic_viz --data-path /home/adam/data/in1k --data-set IMNET --device cuda --batch-size 256 --lr 4e-3 --epochs 400 --weight-decay 0.02 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt lamb --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment
torchrun --nnodes=1 --nproc_per_node=4 main.py
torchrun --nnodes=1 --nproc_per_node=4 main.py --distributed

# ImageNet1k - Setting the token locations dynamically
torchrun --nnodes=1 --nproc_per_node=3 main.py --distributed --num_reg 4 --cls_pos 6 --reg_pos 3

# ImageNet1k + distillation - Setting the token locations dynamically
torchrun --nnodes=1 --nproc_per_node=4 main_distillation.py --distributed --num_reg 4 --cls_pos 6 --reg_pos 3


# CIFAR10 
python main.py --model vit_register_dynamic_viz --data-path /data/CIFAR10/cifar-10-batches-py --data-set CIFAR10 --device cuda --batch-size 256 --lr 4e-3 --epochs 300 --weight-decay 0.05 --sched cosine --input-size 32 --patch-size 4 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 10 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment

# ImageNet1k - in_main.py (simple)
python in_main.py --data-path /home/adam/data/in1k --device cuda --gpu 3 --batch-size 512 --lr 4e-3 --epochs 100 --weight-decay 0.05 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --color-jitter 0.3 --ThreeAugment


# Visualization 
# ImageNet1k - in_main.py (simple)
python viz_attn_in_main.py --cls_pos 0 --reg_pos 0 --layer_num 3 --img_num 0 
