# Uncertainty-Aware-Perceiver

I added Deep Ensemble, Stochastic Weighted Average, Snapshot Ensemble, and MC Dropout on Perceiver (Fourier) to optimize the Perceiver. This repo also contains baselines consisted of Vision Transformer, Resnet, Perceiver(Conv), Perceiver(Learnable PE), and Perceiver(Fourier PE).

Please, refer to ``Makefile`` for example command and refer to ``environment.yaml`` for environment. First argument refers to file name to save model weights, second argument determines if you want to train/evaluate or evaluate your model (1 for train/eval, 0 for eval), third argument refers to dataset you want to use.
