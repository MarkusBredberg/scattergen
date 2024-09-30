import itertools
import os
from utils.data_loader import load_galaxies, get_classes
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from kymatio.torch import Scattering2D
from utils.models import get_model
from utils.custom_mse import NormalisedWeightedMSELoss, CustomMSELoss, WeightedMSELoss, RadialWeightedMSELoss, CustomIntensityWeightedMSELoss, MaxIntensityMSELoss, StandardMSELoss, CombinedMSELoss, ExperimentalMSELoss, BasicMSELoss
from utils.scatter_reduction import lavg, ldiff
from utils.training_tools import EarlyStopping 
from torchsummary import summary
from tqdm import tqdm
import time
from utils.plotting import vae_plot_comparison, loss_vs_epoch, plot_original_images, plot_weight_and_loss_maps, plot_images, plot_histograms
import matplotlib.pyplot as plt


######################################################################################################
################################### CONFIGURATION ####################################################
######################################################################################################

#encoder_list = ['CDual', 'CCNN', 'CSTMLP', 'CldiffSTMLP', 'ClavgSTMLP']
#encoder_list = ['STMLP', 'lavgSTMLP', 'ldiffSTMLP']
#encoder_list = ['CCNN']
#encoder_list = ['Dual', 'CNN']
encoder_list = ['STMLP']

ind = 0 #Choose the loss function to use

#galaxy_classes = [[10, 11, 12, 13]] #Use double square parenthesis for conditional VAEs
galaxy_classes = [11]
num_galaxies_list = [100+ind]
hidden_dim1 = [256]
hidden_dim2 = [128]
latent_dims = [32]
learning_rates = [1e-4]
reg_params = [1e-4] # Regularization parameters
initial_final_betas = [(1e-1, 1)]
num_epochs_cuda = 500; num_epochs_cpu = 10
batch_size = 128 # Adjust batch size to manage memory usage
img_shape = (1, 128, 128)
J, L, order = 3, 12, 2

FFCV = True # Use five-fold cross-validation
ES = True # Use early stopping
IMGCHECK = False # Check the input images (Tool for control)
SAVEIMGS = False # Save the reconstructed images in tensor format
NORMALISETOPM = False # Normalise to [-1, 1]
PLOTFAILED = True # Plot the top 5 images with the largest MSE loss

#########################################################################################################################

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Running script 2.scattervae_training.py")


if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
    print(f"CUDA is available. Setting epochs to {num_epochs}.")
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_cpu
    print(f"CUDA is not available. Setting epochs to {num_epochs}.")

classes = get_classes()

train_loader, test_loader = None, None
for galaxy_class, num_galaxies, encoder_choice, hidden_dim1, hidden_dim2, latent_dim, lr, reg, (initial_beta, final_beta), fold in itertools.product(
        galaxy_classes, num_galaxies_list, encoder_list, hidden_dim1, hidden_dim2, latent_dims, reg_params, learning_rates, initial_final_betas, range(5) if FFCV else [6]):
    print(f"Training {encoder_choice} on {num_galaxies} galaxies of type {galaxy_class} with hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}, latent_dim={latent_dim}, lr={lr}, beta={initial_beta}-{final_beta}, fold={fold}")

    runname = f'{galaxy_class}_{num_galaxies}_{encoder_choice}_{fold}'
    log_path = f"./generator/log_{runname}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file = open(log_path, 'w')

    # Free memory explicitly
    if train_loader and test_loader:
        del train_loader, test_loader
    torch.cuda.empty_cache()

    data = load_galaxies(galaxy_class=galaxy_class, 
                        fold=fold,
                        img_shape=img_shape, 
                        sample_size=num_galaxies, 
                        process=True, 
                        train=True, 
                        runname=None, 
                        generated=False, 
                        reconstructed=False)
    train_images, train_labels, test_images, test_labels = data
    
    # Check the input data
    print("Train images shape before filtering:", np.shape(train_images))
    if IMGCHECK:
        plot_images(train_images, train_labels)
        plot_histograms(train_images, test_images)


    # Prepare input data
    if 'ST' in encoder_choice or 'Dual' in encoder_choice:
        scattering = Scattering2D(J=J, L=L, shape=img_shape[1:], max_order=order).to(DEVICE)    
        def compute_scattering_coeffs(images):
            print("Computing scattering coefficients...")
            scat_batch_size = 8
            start_time = time.time()
            num_images = images.shape[0]
            all_scat_coeffs = []
            for i in range(0, num_images, scat_batch_size):
                batch_images = images[i:i+scat_batch_size].contiguous().to(DEVICE)
                with torch.no_grad():  # Disable gradient calculation
                    batch_scat_coeffs = scattering(batch_images)
                    if batch_scat_coeffs.dim() == 3:
                        batch_scat_coeffs = batch_scat_coeffs.unsqueeze(0)
                    batch_scat_coeffs = torch.squeeze(batch_scat_coeffs)
                    all_scat_coeffs.append(batch_scat_coeffs.to(DEVICE))
            all_scat_coeffs = [coeff if coeff.dim() == all_scat_coeffs[0].dim() else coeff.unsqueeze(0) for coeff in all_scat_coeffs]
            all_scat_coeffs = torch.cat(all_scat_coeffs, dim=0)
            elapsed_time = time.time() - start_time
            print(f"Time taken to compute scattering coefficients: {elapsed_time:.2f} seconds")
            file.write(f"Time taken to compute scattering coefficients: {elapsed_time:.2f} seconds \n")
            return all_scat_coeffs

        train_scat_coeffs = compute_scattering_coeffs(train_images).to(DEVICE)
        test_scat_coeffs = compute_scattering_coeffs(test_images).to(DEVICE)
        scatshape = int(np.prod(train_scat_coeffs.shape[1:]))

    if 'lavg' in encoder_choice or 'ldiff' in encoder_choice:
        if 'lavg' in encoder_choice:
            train_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])
        else:
            train_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])
                
    def normalize_to_0_1(images, batch_size=64):
        normed_images = torch.zeros_like(images)  # Placeholder for the normalized images

        # Split the images into smaller sub-batches
        num_images = images.size(0)
        for i in range(0, num_images, batch_size):
            batch = images[i:i + batch_size]

            # Min and max per sub-batch
            min_vals = batch.view(batch.size(0), -1).min(dim=1, keepdim=True)[0].view(batch.size(0), 1, 1, 1)
            max_vals = batch.view(batch.size(0), -1).max(dim=1, keepdim=True)[0].view(batch.size(0), 1, 1, 1)

            # Normalize to [0, 1]
            normed_batch = (batch - min_vals) / (max_vals - min_vals + 1e-8)
            normed_batch[torch.isnan(normed_batch)] = 0.0  # Set any NaN values to 0
            normed_images[i:i + batch_size] = normed_batch

        return normed_images

    # Normalize each image individually to [-1, 1] after normalizing to [0, 1]
    def normalize_to_minus1_1(images):
        return images * 2 - 1

    # Normalize train and test images to [0, 1]
    train_images = normalize_to_0_1(train_images)
    test_images = normalize_to_0_1(test_images)

    if NORMALISETOPM:
        # If NORMALISETOPM is True, normalize to [-1, 1]
        train_images = normalize_to_minus1_1(train_images)
        test_images = normalize_to_minus1_1(test_images)

    # Handle scattering coefficients normalization in a similar way
    if 'ST' in encoder_choice or 'Dual' in encoder_choice:
        train_scat_coeffs = normalize_to_0_1(train_scat_coeffs)
        test_scat_coeffs = normalize_to_0_1(test_scat_coeffs)

        if NORMALISETOPM:
            train_scat_coeffs = normalize_to_minus1_1(train_scat_coeffs)
            test_scat_coeffs = normalize_to_minus1_1(test_scat_coeffs)

    
    #Check input after renormalisation and filtering  
    if IMGCHECK: 
        plot_images(train_images, train_labels)
        plot_histograms(train_images, test_images)

    if 'ST' in encoder_choice or 'Dual' in encoder_choice: #Double dataset for convenience for dual model in training loop
        train_dataset = TensorDataset(train_scat_coeffs, train_images)
        test_dataset = TensorDataset(test_scat_coeffs, test_images)
    else: 
        train_dataset = TensorDataset(train_images, train_images) 
        test_dataset = TensorDataset(test_images, test_images) 


    # Create the data loaders
    def custom_collate(batch):
        if len(batch) < 3: # Drop the batch if it contains fewer than 3 images
            return None
        return torch.utils.data.dataloader.default_collate(batch)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)


    # Define the model
    num_classes = train_labels.max().item() + 1 
    print(f"train_labels min: {train_labels.min()}, max: {train_labels.max()}, num_classes: {num_classes}")
    if 'C' in encoder_choice and encoder_choice != 'CNN':
        train_labels = torch.nn.functional.one_hot(train_labels.squeeze(), num_classes=num_classes).float()
        test_labels = torch.nn.functional.one_hot(test_labels.squeeze(), num_classes=num_classes).float()
    scatshape = np.shape(train_scat_coeffs)[1:] if "ST" in encoder_choice or "Dual" in encoder_choice else np.array([1, 1, 1])    
    model = get_model(encoder_choice, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes)
    model.to(DEVICE)


    # Print model summary
    if 'C' in encoder_choice and encoder_choice != 'CNN':
        print("Summary not available for conditional VAEs")
    elif 'ST' in encoder_choice:
        summary(model, input_size=train_scat_coeffs[0].shape, device=DEVICE)
    elif 'Dual' in encoder_choice:
        summary(model, input_size=[img_shape, train_scat_coeffs[0].shape], device=DEVICE)      
    else:
        summary(model, input_size=img_shape, device=DEVICE)


    # Define the loss function and optimizer
    def lr_lambda(epoch):
        if epoch < num_epochs * 0.05:
            return epoch / (num_epochs * 0.05)
        return 0.5 * (1 + np.cos(np.pi * (epoch - num_epochs * 0.05) / (num_epochs * 0.95)))
    model.apply(lambda m: nn.init.calculate_gain('leaky_relu', 0.2) if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)) else None)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=reg)
    #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    beta_increment = (final_beta - initial_beta) / num_epochs
    
    MSEfun = ['blablabla', 'normalised', 'radial', 'weighted', 'intensity', 'maxintensity', 'custom', 'combined', 'experimental', 'basic'][ind]
    print("Ind: ", ind, " and MSEfun: ", MSEfun)
    if MSEfun == 'radial': mse_loss = RadialWeightedMSELoss(threshold=0.1, intensity_weight=0.001, radial_weight=0.001)
    elif MSEfun == 'normalised': mse_loss = NormalisedWeightedMSELoss(threshold=0.1, weight=0.5) 
    elif MSEfun == 'weighted': mse_loss = WeightedMSELoss(threshold=0.1, weight=0.001)
    elif MSEfun == 'intensity': mse_loss = CustomIntensityWeightedMSELoss(intensity_threshold=0.07, intensity_weight=0.0001, log_weight=0.0001)
    elif MSEfun == 'maxintensity': mse_loss = MaxIntensityMSELoss(intensity_weight=0.001)
    elif MSEfun == 'standard': mse_loss = StandardMSELoss()
    elif MSEfun == 'custom': mse_loss = CustomMSELoss(intensity_weight=0.001, sum_weight=0.001)
    elif MSEfun == 'combined': mse_loss = CombinedMSELoss(intensity_weight=0.001, sum_weight=0.001, threshold=0.1, high_intensity_weight=0.001)
    elif MSEfun == 'experimental': mse_loss = ExperimentalMSELoss(intensity_weight=0.001, sum_weight=0.001, histogram_weight=0.001, threshold=0.5, high_intensity_weight=0.001)
    elif MSEfun == 'basic': mse_loss = BasicMSELoss(threshold=0)
    else: mse_loss = nn.MSELoss(reduction='sum')
    
    def vae_loss_function(x, x_hat, mean, log_var, beta=1.0):
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        if x_hat.shape != x.shape:
            x_hat = x_hat.view(x.shape)
        RecLoss = mse_loss(x_hat, x)
        return RecLoss + beta * KLD


    # Train the model
    model.train()
    train_losses, val_losses = [], []
    train_mse, train_kl = [], []
    val_mse, val_kl = [], []
    best_loss, best_model_state = float('inf'), None
    early_stopping = EarlyStopping(patience=25)
    model_path = f'./generator/model_{runname}.pth'
    start_time = time.time()
    with tqdm(total=num_epochs, desc=f"Training {runname}", position=0) as pbar:
        for epoch in range(num_epochs):
            
            # Training
            beta = initial_beta + epoch * beta_increment
            overall_loss = 0
            epoch_mse, epoch_kl = 0, 0
            with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}", leave=False, position=1) as epoch_bar:
                for batch_idx, batch in epoch_bar:
                    if batch is None:
                        continue  
                    scat_coeffs, x = batch  
                    x = x.to(DEVICE)
                    
                    optimizer.zero_grad()

                    if 'ST' in encoder_choice or 'Dual' in encoder_choice: # Scattering transform
                        scat_coeffs = scat_coeffs.to(DEVICE)
                        if 'C' in encoder_choice: # Conditional VAE
                            labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                            if 'MLP' in encoder_choice:
                                x_hat, mean, log_var = model(scat_coeffs, labels)
                            elif 'CDual' in encoder_choice:
                                x_hat, mean, log_var = model(x, scat_coeffs, labels)
                        elif 'Dual' in encoder_choice:
                            x_hat, mean, log_var = model(x, scat_coeffs)
                        else:
                            x_hat, mean, log_var = model(scat_coeffs)
                    elif 'C' in encoder_choice and encoder_choice != 'CNN': # Conditional VAE - No scattering transform
                        labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                        x_hat, mean, log_var = model(x, labels)
                    else:
                        x_hat, mean, log_var = model(x)
                        
                    if x_hat.shape != x.shape:
                        x_hat = x_hat.view(x.shape)

                    loss = vae_loss_function(x, x_hat, mean, log_var, beta=beta)
                    mse = mse_loss(x_hat, x).item()
                    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()).item()
                    overall_loss += loss.item()
                    epoch_mse += mse
                    epoch_kl += kl
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                    #scheduler.step()
                    epoch_bar.set_description(f"Epoch {epoch + 1} Batch {batch_idx + 1} Loss: {loss.item():.4f}")

            average_loss = overall_loss / len(train_loader.dataset)
            epoch_mse /= len(train_loader.dataset)
            epoch_kl /= len(train_loader.dataset)
            train_losses.append(average_loss)
            train_mse.append(epoch_mse)
            train_kl.append(epoch_kl)

            if average_loss < best_loss or epoch == 0:
                best_loss = average_loss
                best_model_state = model.state_dict()
                
            # Validation
            model.eval()
            val_loss = 0
            val_mse_epoch, val_kl_epoch = 0, 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if batch is None:
                        continue 
                    scat_coeffs, x = batch  
                    x = x.to(DEVICE)
                    
                    if 'ST' in encoder_choice or 'Dual' in encoder_choice: # Scattering transform
                        scat_coeffs = scat_coeffs.to(DEVICE)
                        if 'C' in encoder_choice: # Conditional VAE
                            labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                            if 'MLP' in encoder_choice:
                                x_hat, mean, log_var = model(scat_coeffs, labels)
                            elif 'CDual' in encoder_choice:
                                x_hat, mean, log_var = model(x, scat_coeffs, labels)
                        elif 'Dual' in encoder_choice:
                            x_hat, mean, log_var = model(x, scat_coeffs)
                        else:
                            x_hat, mean, log_var = model(scat_coeffs)
                    elif 'C' in encoder_choice and encoder_choice != 'CNN': # Conditional VAE - No scattering transform
                        labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                        x_hat, mean, log_var = model(x, labels)
                    else:
                        x_hat, mean, log_var = model(x)

                    if x_hat.shape != x.shape:
                        x_hat = x_hat.view(x.shape)
                        
                    loss = vae_loss_function(x, x_hat, mean, log_var, beta=beta)
                    mse = mse_loss(x_hat, x)
                    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()).item()
                    val_loss += loss.item()
                    val_mse_epoch += mse
                    val_kl_epoch += kl
                val_loss /= len(test_loader.dataset)
                val_mse_epoch /= len(test_loader.dataset)
                val_kl_epoch /= len(test_loader.dataset)
                val_losses.append(val_loss)
                val_mse.append(val_mse_epoch)
                val_kl.append(val_kl_epoch)
                
            model.train()
            pbar.update(1)
            pbar.set_postfix_str(f"Average Loss: {average_loss:.4f}, Validation Loss: {val_loss:.4f}")
            early_stopping(val_loss, model, model_path) # Save the model if validation loss decreases
            if ES:
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                if average_loss < best_loss:
                    best_loss = average_loss
                    best_model_state = model.state_dict()
                
    elapsed_time = time.time() - start_time
    file.write(f"Time taken to train the model: {elapsed_time:.2f} seconds \n")
    file.write(f"Total epochs run before early stopping: {epoch}\n")
    file.write(f"Training Loss: {average_loss:.4f}, Validation Loss: {val_loss:.4f}, ")
    file.write(f"Training MSE: {epoch_mse:.4f}, Training KL: {epoch_kl:.4f}, ")
    file.write(f"Validation MSE: {val_mse_epoch:.4f}, Validation KL: {val_kl_epoch:.4f}\n")


    # Evaluate the model
    model.load_state_dict(best_model_state) # Load the best model
    model.eval()
    reconstructed_images_list, original_images_list = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if batch is None:
                continue 
            scat_coeffs, x = batch
            x = x.to(DEVICE)
            if 'ST' in encoder_choice or 'Dual' in encoder_choice: # Scattering transform
                scat_coeffs = scat_coeffs.to(DEVICE)
                if 'C' in encoder_choice: # Conditional VAE
                    labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                    if 'MLP' in encoder_choice:
                        x_hat, mean, log_var = model(scat_coeffs, labels)
                    elif 'CDual' in encoder_choice:
                        x_hat, mean, log_var = model(x, scat_coeffs, labels)
                elif 'Dual' in encoder_choice:
                    x_hat, mean, log_var = model(x, scat_coeffs)
                else:
                    x_hat, mean, log_var = model(scat_coeffs)
            elif 'C' in encoder_choice and encoder_choice != 'CNN': # Conditional VAE - No scattering transform
                labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size].to(DEVICE)
                x_hat, mean, log_var = model(x, labels)
            else:
                x_hat, mean, log_var = model(x)

            # Verify and append only if shapes match
            if x_hat.shape == x.shape:
                reconstructed_images_list.append(x_hat.cpu())
                original_images_list.append(x.cpu())
            else:
                print(f"Shape mismatch: x_hat shape: {x_hat.shape}, x shape: {x.shape}")
            
            # Save the first batch of reconstructed images, as well as the loss map
            if batch_idx == 0:
                if MSEfun in ['radial', 'normalised', 'intensity', 'weighted', 'maxintensity', 'standard', 'custom', 'combined']:
                    loss_map, weight_map = mse_loss(x_hat, x, return_map=True)
                    loss_map = loss_map.cpu().numpy()
                    loss_map[loss_map == 0] = np.nan
                    plot_weight_and_loss_maps(weight_map, loss_map, x, x_hat, savepath=f'./generator/VAE_{runname}_lossmap.png') 
                save_image(x_hat.view(batch_size, *img_shape), f'./generator/VAE_{runname}_reconstructed_batch0.png') # Save the first batch of reconstructed images
                
    all_reconstructed_images = torch.cat(reconstructed_images_list)
    all_original_images = torch.cat(original_images_list)
    if PLOTFAILED:
        mse_losses = []
        for orig, recon in zip(all_original_images, all_reconstructed_images):
            mse_loss_per_image = torch.mean((orig - recon) ** 2).item()
            mse_losses.append(mse_loss_per_image)
        
        # Get the indices of the five largest MSE losses
        top_5_mse_indices = np.argsort(mse_losses)[-5:]

        # Get the images corresponding to the largest MSE losses
        top_5_original_images = all_original_images[top_5_mse_indices]
        top_5_reconstructed_images = all_reconstructed_images[top_5_mse_indices]
        
        def plot_top_mse_images(originals, reconstructions, mse_losses, num_images=5):
            fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
            for i in range(num_images):
                # Original images
                axs[i, 0].imshow(originals[i].squeeze().cpu().numpy(), cmap='gray')
                axs[i, 0].set_title(f'Original Image {i+1}')

                # Reconstructed images
                axs[i, 1].imshow(reconstructions[i].squeeze().cpu().numpy(), cmap='gray')
                axs[i, 1].set_title(f'Reconstructed Image {i+1}\nMSE Loss: {mse_losses[i]:.4f}')

                # Hide axis for better visibility
                axs[i, 0].axis('off')
                axs[i, 1].axis('off')
            plt.tight_layout()
            plt.savefig(f'./generator/VAE_{runname}_failed_reconstructions.png', bbox_inches='tight')

        # Plot the top 5 images with the largest MSE loss
        plot_top_mse_images(top_5_original_images, top_5_reconstructed_images, np.array(mse_losses)[top_5_mse_indices])


    if SAVEIMGS:
        torch.save(all_reconstructed_images, f'./generator/VAE_{runname}_reconstructed_images.pt')
        print(f'All reconstructed images saved in tensor format with shape: {all_reconstructed_images.shape}')


    # Quick evaluation
    supertitle = f"VAE with {encoder_choice} encoder, hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}, latent_dim={latent_dim}, \n for {num_galaxies} galaxies of type {galaxy_class} and {num_epochs} epochs."
    file.write(f"Supertitle: {supertitle}\n")
    file.write(f"img_shape: {img_shape}\n")
    file.write(f"J: {J}\n")
    file.write(f"L: {L}\n")
    file.write(f"order: {order}\n")
    file.write(f"batch_size: {batch_size}\n")
    file.write(f"learning rate: {lr}\n")
    file.close()
        
    loss_vs_epoch(train_losses, val_losses, save=True, save_path=f"./generator/VAE_{runname}_loss.png")
    x_reshaped = all_original_images.view(-1, img_shape[-2], img_shape[-1]).detach().cpu()
    x_hat_reshaped = all_reconstructed_images.view(-1, img_shape[-2], img_shape[-1]).detach().cpu()

    # Reconstruction comparison
    vae_plot_comparison(x_reshaped[:min(len(x_reshaped), 5)], x_hat_reshaped[:min(len(x_hat_reshaped), 5)], supertitle=supertitle, num_images=min(len(x_reshaped), 5), save=True, save_path=f"./generator/VAE_{runname}_comparison.png")

    # Generate new images and plot
    with torch.no_grad():
        noise = torch.randn((batch_size, latent_dim)).to(DEVICE)
        if 'C' in encoder_choice and encoder_choice != 'CNN':
            sample_labels = torch.arange(0, batch_size) % train_labels.size(1)
            sample_labels = sample_labels.to(DEVICE)
            print("Sample labels:", sample_labels)
            sample_labels_one_hot = torch.nn.functional.one_hot(sample_labels, num_classes=train_labels.size(1)).float()
            generated_images = model.decoder(noise, sample_labels_one_hot)
        else:
            generated_images = model.decoder(noise)       
        save_image(generated_images.view(batch_size, *img_shape), f'./generator/VAE_{runname}_generated_vaesample.png')

    train_loader = DataLoader(TensorDataset(train_images, train_images), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    plot_original_images(train_loader, img_shape[-1], 10, save_path=f'./generator/VAE_{runname}_originals.png')
