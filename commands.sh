# ImageNet1k
python main.py --model vit_register_dynamic_viz --data-path /home/adam/data/in1k --data-set IMNET --batch-size 256 --lr 4e-3 --epochs 800 --weight-decay 0.05 --sched cosine --input-size 224 --patch-size 16 --eval-crop-ratio 1.0 --reprob 0.0 --nodes 1 --ngpus 8 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt fusedlamb --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss --color-jitter 0.3 --ThreeAugment

# CIFAR10 
python main.py --model vit_register_dynamic_viz --data-path /home/adam/data/cifar-10-batches-py --data-set CIFAR --batch-size 256 --lr 4e-3 --epochs 800 --weight-decay 0.05 --sched cosine --input-size 32 --patch-size 4 --eval-crop-ratio 1.0 --reprob 0.0 --nodes 1 --ngpus 8 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 10 --seed 0 --opt fusedlamb --warmup-lr 1e-6 --mixup 0.8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss --color-jitter 0.3 --ThreeAugment
