# Extended Quantum Autoencoder for Image Denoising

![Image output showing the QCAE denoising larger color images depicting complex scenes](https://github.com/jacx8/qcae-gpu-accel/blob/main/denoised_inputs.png)

## Introduction
This is an implementation of a quantum autoencoder for image denoising. The autoencoder is trained on a set of images with added noise. The autoencoder is then used to denoise the same set of images. This repository builds upon the work of [Kea et al.](https://arxiv.org/pdf/2401.06367) The maintainers of this repository chose to extend their work as their final project for their graduate quantum computing class. Our extensions include adding support for larger image sizes, color images, and enabling GPU acceleration for faster encoding, decoding, and simulation times. The focus of our work was to assess the potential for this model to outperform classical convolutional autoencoders in practical real-world image denoising tasks. Toward this end, we tested the model with Fashion MNIST and CIFAR-10 datasets after augmenting the model with the appropriate extensions.

### Autoencoder
There are 3 main components of the autoencoder:
- Encoder
- Latent space: The latent space is the space between the encoder and decoder. It is a compressed representation of the input image.
- Decoder

### Quantum Autoencoder
Due to the limitations of latent space in a classical autoencoder, a quantum approximate optimization algorithm is used to find the optimal latent space.

## Running the system on Palmetto 2

### Requesting an appropriate node allocation
```bash
salloc --nodes 1 --ntasks-per-node 1 --cpus-per-task 2 --mem 8G --time 03:00:00 -C interconnect_hdr,chip_type_6148g --gpus-per-node v100:1
```

### Environment setup
```bash
module load anaconda3
conda create -n qcae_env python=3.10.16
source activate qcae_env
pip install -r requirements.text
python3 -W "ignore" main.py
```

### Training
```bash
python3 -W "ignore" main.py
```

### Testing
```bash
python3 -W "ignore" main.py --test
```

