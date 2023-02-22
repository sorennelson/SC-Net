# SC-Net
Code for the paper [Bijective-constrained cycle-consistent deep learning for optics-free imaging and classification](https://opg.optica.org/optica/fulltext.cfm?uri=optica-9-1-26&id=466316)

## Setup
1. Upload data to 'datasets' folder. Assumes lensless images are jpegs with names matching the CIFAR-10 images. I have the following file structure:
    - **Lensless images**: `./datasets/lensless-cifar/20x20-cifar10/xxx.jpeg` and `./datasets/lensless-cifar/20x20-cifar10/test-xxx.jpeg` for test images.
    - **CIFAR-10 images**: `./datasets/lensless-cifar/cifar10/data_batch_x`. 
    If you use a different file structure you will need to edit Data.py `setup_datasets()` and `PARAMS['dir']` inside Train.py.
2. Create directory 'figures'

### Required Packages 
If no version is specified any should work
- Tensorflow (version 2.3.0)
- Tensorflow Addons (version 0.11.2)
- Numpy
- Matplotlib
- Scikit-image

## Training
1. Configure Train.py PARAMS.
2. Run `python Train.py`

Main results are with following PARAMS configuration:
```
PARAMS = {
    'change': '',
    'dir': 'lensless-cifar', 'num_channels': 3, 'raw_input_shape': (180,240), 'target_input_shape': (32,32),

    'generator': cycle_reconstructor,
    'rec_loss': dist_mae_loss,
    'cycle_loss': dist_mae_loss,
    'chpt': get_and_increment_chpt_num(),
    'channel_format': 'NCHW',
    'latent': True,

    'load_chpt': None,
    'save_models': True,

    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 2e-4,
    'disc_lr': 2e-5,
    
     # Loss scalars:
        # type double: Loss = GAN + beta(MAE + lambda(Forward + lambda_b*Backward))
    'lambda': 1,
    'lambda_b': 0.1,
    'gamma': 0,
    'beta': 100,

    'linear_decay_lr': None, # None for no decay, or integer for number of epochs to start decaying after
    'generator_to_discriminator_steps': 0, # 0: update both each step, 1: 1 generator step then 1 discriminator, 2: 2 generator steps then 1 discriminator, ...

    'type': 'double', # Self Consistent Supervised

    'F_PARAMS': {
        'filters': {'down': [64, 128, 256, 256], 'up': [256, 128, 64]},
        'dropout': {'down': [0.0, 0.0, 0.0, 0.0], 'up': [0.0, 0.0, 0.0]},
        'kernels': [5,5], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'batch',
        'activation': 'relu',
        'compression': True,
    },
    'G_PARAMS': {
        'filters': {'down':[64, 128, 256, 256], 'up':[256, 128, 64]},
        'dropout': {'down':[0.0, 0.0, 0.0, 0.0], 'up':[0.0, 0.0, 0.0]},
        'kernels': [5,5], 'dilation_rate': 2,
        'res_depth': {'down':1, 'bottom':2, 'up':1},
        'norm': 'batch',
        'activation': 'relu',
        'compression': True,
    },
}
```

## Image Generation and Evaluation from presaved G Network
1. Configure Test_and_generate.py PARAMS.
2. Run `python Test_and_generate.py`
