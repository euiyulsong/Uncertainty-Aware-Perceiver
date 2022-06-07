perceiver:
	TORCH_CUDA_ARCH_LIST=8.4 CUDA_VISIBLE_DEVICES=2 nohup python perceiver.py perceiver 1 cifar10 > perceiver.log &
perceiver_conv:
	TORCH_CUDA_ARCH_LIST=8.4 CUDA_VISIBLE_DEVICES=2 nohup python perceiver_conv.py perceiver_conv 1 cifar10 > perceiver_conv.log &
perceiver_f:
	TORCH_CUDA_ARCH_LIST=8.4 CUDA_VISIBLE_DEVICES=2 nohup python perceiver_fourier.py perceiver_fourier 1 cifar10 > perceiver_fourier.log &
fast:
	TORCH_CUDA_ARCH_LIST=8.4 CUDA_VISIBLE_DEVICES=1 nohup python fast.py fast 1 cifar100 > fast.log &
deep:
	TORCH_CUDA_ARCH_LIST=8.4 CUDA_VISIBLE_DEVICES=7 nohup python deep_ensemble.py deep 1 > deep.log &
mc_dropout:
	TORCH_CUDA_ARCH_LIST=8.4 CUDA_VISIBLE_DEVICES=0 nohup python mc_dropout.py mc_dropout 1 cifar10 > mc_dropout.log &
snap:
	TORCH_CUDA_ARCH_LIST=8.4 CUDA_VISIBLE_DEVICES=6 nohup python snap.py snap 1 cifar10 > snap.log &
swa:
	TORCH_CUDA_ARCH_LIST=8.4 CUDA_VISIBLE_DEVICES=6 nohup python swa.py swa 1 cifar10 > swa.log &
vit:
	TORCH_CUDA_ARCH_LIST=8.4 CUDA_VISIBLE_DEVICES=6 nohup python vit.py vit 1  > vit.log
