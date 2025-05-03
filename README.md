# Quantum Autoencoder for Image Denoising

![Image output showing the QCAE denoising larger color images depicting complex scenes](https://github.com/jacx8/qcae-gpu-accel/blob/main/denoised_inputs.png)
![Image output showing the QCAE loss for the above output](https://github.com/jacx8/qcae-gpu-accel/blob/main/losses.png)

## Introduction
This is an implementation of a quantum autoencoder for image denoising. The autoencoder is trained on a set of images with added noise. The autoencoder is then used to denoise the same set of images.

### Autoencoder
There are 3 main components of the autoencoder:
- Encoder
- Latent space: The latent space is the space between the encoder and decoder. It is a compressed representation of the input image.
- Decoder

### Quantum Autoencoder
Due to the limitations of latent space in a classical autoencoder, a quantum approximate optimization algorithm is used to find the optimal latent space.

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

## Usage

Before proceeding, replace the IBM token in the `circuit.py` file.

### Training
```bash
python main.py
```

### Testing
```bash
python main.py --test
```

