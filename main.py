import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Function
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

from hybrid_conv import ConvDenoiseNet
from datasets import train_dataset, test_dataset

from constant import *
from add_noise import add_gaussian_noise
import argparse

from qiskit_ibm_provider import IBMProvider

from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
    args = parser.parse_args()
    

    train_loader = train_dataset(n_samples=10)
    test_loader = test_dataset(n_samples=10)

    if not args.test:
        # Create model
        model = ConvDenoiseNet()
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        # number of epochs to train the model
        n_epochs = 300

        losses = []
        psnrs = []

        for epoch in tqdm(range(1, n_epochs + 1)):
            train_loss = 0.0
            for data in train_loader:
                images, _ = data
                images = images.to(device, non_blocking=True) 
                noisy_imgs = add_gaussian_noise(images)
                optimizer.zero_grad()
                outputs = model(noisy_imgs)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

                outputs = outputs.detach().view(len(images), NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
                batch_avg_psnr = 0
                for i in range(len(images)):
                    #org = np.transpose(images[i], (1, 2, 0)).detach().numpy()
                    org = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
                    #denoise = np.transpose(outputs[i], (1, 2, 0)).detach().numpy()
                    denoise = np.transpose(outputs[i].detach().cpu().numpy(), (1, 2, 0))
                    batch_avg_psnr += psnr(org, denoise)

            psnrs.append(batch_avg_psnr)

            train_loss = train_loss/len(train_loader)
            losses.append(train_loss)

        print('PSNRs: ', psnrs)

        # Plot losses
        plt.figure()
        plt.plot(losses)
        plt.savefig('losses.png')
        np.savetxt("losses.csv", np.array(losses), delimiter=",")

        # Plot PSNRs
        plt.figure()
        plt.plot(psnrs)
        plt.savefig('psnrs.png')
        np.savetxt("psnrs.csv", np.array(psnrs), delimiter=",")

        # Save the model
        torch.save(model.state_dict(), "model.pt")
    else:
        # Load model
        model = ConvDenoiseNet()
        model = ConvDenoiseNet().to(device)
        model.load_state_dict(torch.load("model.pt"))

    # plot the first ten input images and then reconstructed images
    ncols = 10
    fig, axes = plt.subplots(nrows=3, ncols=ncols, sharex=True, sharey=True, figsize=(25, 7))
    # fig.tight_layout()

    model.eval()
    with torch.no_grad():
        dataiter = iter(test_loader)
        for k in range(ncols):
            images, labels = next(dataiter)
            images     = images.to(device, non_blocking=True)
            noisy_imgs = add_gaussian_noise(images, 0.25)
            output = model(noisy_imgs)

            output = output.view(1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            
            images_np = images.squeeze().cpu().numpy()
            noisy_imgs_np = noisy_imgs.squeeze().cpu().numpy()
            output_np = output.squeeze().cpu().numpy()

            if NUM_CHANNELS == 3:                                                                                                                                                                       
                images_np = np.transpose(images_np, (1, 2, 0))
                noisy_imgs_np = np.transpose(noisy_imgs_np, (1, 2, 0))
                output_np = np.transpose(output_np, (1, 2, 0))
 
            # Plot
            col_axes = axes[:, k]
            if NUM_CHANNELS == 1:
                col_axes[0].imshow(images_np, cmap='gist_gray')        # Original
                col_axes[1].imshow(noisy_imgs_np, cmap='gist_gray')     # Noisy
                col_axes[2].imshow(output_np, cmap='gist_gray')         # Output
            else:
                col_axes[0].imshow(images_np)        # Original
                col_axes[1].imshow(noisy_imgs_np)     # Noisy
                col_axes[2].imshow(output_np)         # Output


            col_axes = axes[:, k]
            #col_axes[0].imshow(np.squeeze(images), cmap='gist_gray')
            col_axes[0].imshow(images_np, cmap='gist_gray')
            #col_axes[1].imshow(np.squeeze(noisy_imgs), cmap='gist_gray')
            col_axes[1].imshow(noisy_imgs_np, cmap='gist_gray')
            #col_axes[2].imshow(np.squeeze(output), cmap='gist_gray')
            col_axes[2].imshow(output_np, cmap='gist_gray')

        plt.savefig('denoised_inputs.png', dpi=400)


    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images = data[0].to(device, non_blocking=True)

            noisy_imgs = add_gaussian_noise(images, sigma=1)
            output = model(noisy_imgs)
            output = output.view(len(images), NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            output = output.detach().cpu()

            images = images.cpu().numpy().squeeze()
            output = output.cpu().numpy().squeeze()
            
            img1 = images[0]   # grab the first (and only) image
            img2 = output[0]   # grab the first (and only) output

            if NUM_CHANNELS == 3:
                ssim_val = ssim(img1, img2, data_range=1.0, channel_axis=0)
            else:
                ssim_val = ssim(img1, img2, data_range=1.0)

    print("SSIM: ", ssim_val)


if __name__ == "__main__":
    main()
