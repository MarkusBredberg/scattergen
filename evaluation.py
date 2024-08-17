import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from kymatio.torch import Scattering2D
from utils.data_loader import load_galaxies, get_classes
from utils.scatter_reduction import lavg, ldiff
from utils.models import get_model
from utils.plotting import vae_multmod
from sklearn.metrics import mean_squared_error, silhouette_score
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from scipy.linalg import sqrtm
from numpy import cov, trace, iscomplexobj
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torchvision
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models import inception_v3
import time

########################################################################################################
################################### CONFIGURATION ######################################################
########################################################################################################

# Configuration for auto-loading models
galaxy_classes, num_galaxies = 11, 10007
hidden_dim1, hidden_dim2, latent_dim = 256, 128, 50
J, L, order = 4, 6, 2
num_display, num_generate = 5, 1000

include_two_point_correlation = False
FFCV = True # Use five-fold cross-validation
NORMALISETOPM = False # Normalise to [-1, 1]

num_gen = num_generate * 5 if FFCV else num_generate

models = None # if autoloading
# Paths to the specific models instead of auto-loading
# models = [{"name": "ldiffSTMLP", "path": "./generator/model_15_2000_ldiffSTMLP_320_50_0.001_0.01_0.5.pth", "hidden_dim": 320, "latent_dim": 50},
#    {"name": "ldiffSTMLP", "path": "./generator/model_15_300_ldiffSTMLP_320_64_0.003_0.01_0.5.pth", "hidden_dim": 320, "latent_dim": 64}]

encoder_list = ['Alex', 'CNN', 'Dual', 'STMLP', 'ldiffSTMLP', 'lavgSTMLP']
#encoder_list = ['Dual', 'STMLP']
img_shape = (1, 128, 128)




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = get_classes()
num_classes = len(galaxy_classes) if isinstance(galaxy_classes, list) else 1

print("Running script 3.scattervae_eval.py")

# Create save directory for the combined classes
save_dir = f'./generator/eval/{galaxy_classes}_last_evaluation'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load the data
data = load_galaxies(galaxy_class=galaxy_classes, fold=5, img_shape=img_shape, sample_size=num_galaxies, process=False, train=True)
train_images, train_labels, test_images, test_labels = data
print("Shape of trian images: ", train_images.shape)
print("Shape of test images: ", test_images.shape)

# Calculate scattering coefficients
print("Calculating scattering coefficients")
scattering = Scattering2D(J=J, L=L, shape=img_shape[1:], max_order=order).to(DEVICE)
scat_coeffs = torch.squeeze(scattering(test_images.contiguous().to(DEVICE)))
lavg_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in scat_coeffs])
ldiff_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in scat_coeffs])

if NORMALISETOPM:
    scat_coeffs = (scat_coeffs - torch.mean(scat_coeffs, dim=0)) / torch.maximum(torch.max(torch.abs(scat_coeffs)), torch.tensor(1e-8).to(scat_coeffs.device))
    lavg_scat_coeffs = (lavg_scat_coeffs - torch.mean(lavg_scat_coeffs, dim=0)) / torch.maximum(torch.max(torch.abs(lavg_scat_coeffs)), torch.tensor(1e-8).to(lavg_scat_coeffs.device))
    ldiff_scat_coeffs = (ldiff_scat_coeffs - torch.mean(ldiff_scat_coeffs, dim=0)) /torch.maximum(torch.max(torch.abs(ldiff_scat_coeffs)), torch.tensor(1e-8).to(ldiff_scat_coeffs.device))
    test_images = test_images*2-1 
else:
    scat_coeffs = scat_coeffs / scat_coeffs.max() #normalise to [0, 1] instead
    lavg_scat_coeffs = lavg_scat_coeffs / lavg_scat_coeffs.max()
    ldiff_scat_coeffs = ldiff_scat_coeffs / ldiff_scat_coeffs.max()


# Calculate the input dimensions
scatshape = np.shape(scat_coeffs)
lavg_scatshape = np.shape(lavg_scat_coeffs)
ldiff_scatshape = np.shape(ldiff_scat_coeffs)




###########################################################
################# EVALUATION FUNCTIONS #####################
###########################################################


try:
    # For torchvision version 0.10.0 and later
    inception_weights = torchvision.models.Inception_V3_Weights.DEFAULT
    inception_model = inception_v3(weights=inception_weights)
except AttributeError:
    # For older versions of torchvision
    inception_model = inception_v3(pretrained=True)

inception_model.fc = torch.nn.Identity()  # Use the model up to the last pooling layer
inception_model.eval()
inception_model.to(DEVICE)

def extract_features(images):
    features = []
    with torch.no_grad():
        for img in images:
            img = img.cpu()
            if img.shape[0] == 1:  # Single-channel image
                img = img.repeat(3, 1, 1)  # Repeat the single channel three times to create a 3-channel image
            img = img.unsqueeze(0).to(DEVICE)  # Add batch dimension
            feature = inception_model(img)
            features.append(feature.cpu().numpy())
    features = np.concatenate(features, axis=0)
    return features




def calculate_fid(real_features, generated_features):
    # Ensure there are no NaNs or Infs in the features
    if np.any(np.isnan(real_features)) or np.any(np.isnan(generated_features)) or np.any(np.isinf(real_features)) or np.any(np.isinf(generated_features)):
        raise ValueError("Features contain NaN or Inf values.")
    
    mu1, sigma1 = real_features.mean(axis=0), cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_model_name(galaxy_classes, num_galaxies, encoder, fold):
    if isinstance(galaxy_classes, int):
        runname = f'{galaxy_classes}_{num_galaxies}_{encoder}_{fold}'
        return f'./generator/model_{runname}.pth'
    else:
        galaxy_classes_str = ", ".join(map(str, galaxy_classes))
        runname = f'[{galaxy_classes_str}]_{num_galaxies}_{encoder}_{fold}'
        return f'./generator/model_{runname}.pth'


def load_model(path, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes):
    print(f"Loading model from {path}")
    name = path.split('_')[-2].split('.')[0]
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model = get_model(name=name, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes)


    # Adapt the state_dict to match model's state_dict
    model_state_dict = model.state_dict()
    adapted_state_dict = {}
    for key in model_state_dict.keys():
        if key in state_dict and model_state_dict[key].shape == state_dict[key].shape:
            adapted_state_dict[key] = state_dict[key]
        else:
            print(f"Skipping {key} due to size mismatch or missing key")

    # Load the adapted state dict
    model.load_state_dict(adapted_state_dict, strict=False)
    model.to(DEVICE)
    return model


def compute_silhouette_score(latent_space, n_clusters=2):
    # Replace NaN values with zeros
    latent_space = np.nan_to_num(latent_space)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(latent_space)
    
    # Check if the number of unique labels is greater than 1 and if the number of samples is sufficient
    if len(np.unique(labels)) > 1 and latent_space.shape[0] > 2:
        silhouette = silhouette_score(latent_space, labels)
    else:
        silhouette = float('nan')  # Assign NaN if the silhouette score cannot be computed

    return silhouette



def reconstruct_and_evaluate(model, images, scat_coeffs, lavg_scat_coeffs, ldiff_scat_coeffs, labels, path, batch_size=32):
    print(f"Reconstructing images using {path}")
    if 'lavg' in path:
        scat_coeffs = lavg_scat_coeffs
    elif 'ldiff' in path:
        scat_coeffs = ldiff_scat_coeffs
    
    model.eval()
    all_reconstructed_images = []
    all_latent_representations = []
    mse_list = []
    kl_divergence_list = []
    silhouette_scores = []

    with torch.no_grad():
        num_batches = (len(images) + batch_size - 1) // batch_size

        for i in range(num_batches):
            x_batch = images[i*batch_size:(i+1)*batch_size].to(DEVICE)
            scat_batch = scat_coeffs[i*batch_size:(i+1)*batch_size].to(DEVICE) if scat_coeffs is not None else None
            labels_batch = labels[i*batch_size:(i+1)*batch_size].to(DEVICE) if labels is not None else None

            if 'ST' in path or 'Dual' in path:
                if 'C' in path and 'MLP' in path:
                    x_hat, mean, log_var = model(scat_batch, labels_batch)
                elif 'Dual' in path:
                    x_hat, mean, log_var = model(x_batch, scat_batch)
                else:
                    x_hat, mean, log_var = model(scat_batch)
            elif 'C' in path and 'CNN' not in path:
                x_hat, mean, log_var = model(x_batch, labels_batch)
            else:
                x_hat, mean, log_var = model(x_batch)
            
            reconstructed_images = x_hat.view(-1, *img_shape)
            all_reconstructed_images.append(reconstructed_images.cpu())
            
            z = model.reparameterize(mean, log_var)
            all_latent_representations.append(z.cpu())

            images_flat = x_batch.view(x_batch.size(0), -1).cpu().numpy()
            reconstructed_images_flat = reconstructed_images.view(reconstructed_images.size(0), -1).cpu().numpy()
            mse = np.mean((images_flat - reconstructed_images_flat) ** 2, axis=1)
            mse_list.append(mse)

            if mean is not None and log_var is not None:
                kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1).cpu().numpy()
                kl_divergence_list.append(kl_divergence)
            
            z_np = z.cpu().numpy()
            if z_np.shape[0] > 2:  # Only calculate silhouette score if there are more than 2 samples
                silhouette = compute_silhouette_score(z_np, n_clusters=2)
            else:
                silhouette = float('nan')  # Skip silhouette score calculation if not enough samples
            silhouette_scores.extend([silhouette] * x_batch.size(0))

    all_reconstructed_images = torch.cat(all_reconstructed_images, dim=0)
    all_latent_representations = torch.cat(all_latent_representations, dim=0)
    mse = np.concatenate(mse_list, axis=0)
    kl_divergence = np.concatenate(kl_divergence_list, axis=0) if kl_divergence_list else None
    silhouette_scores = np.array(silhouette_scores)
    
    mean_mse = np.mean(mse)
    mean_kl_divergence = np.mean(kl_divergence) if kl_divergence is not None else None
    mean_silhouette = np.nanmean(silhouette_scores)  # Handle NaN values appropriately

    return all_reconstructed_images, all_latent_representations, mean_mse, mean_kl_divergence, mean_silhouette


def generate_from_noise(model, latent_dim, num_samples=5):
    print("Generating images from noise")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        noise = torch.randn(num_samples, latent_dim).to(DEVICE)
        if hasattr(model, 'decoder') and 'conditional' in str(type(model.decoder)).lower():
            labels = torch.ones(num_samples, dtype=torch.long).to(DEVICE)  # Assuming label '0'
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()  # One-hot encode the labels
            generated_images = model.decoder(noise, labels)
        else:
            generated_images = model.decoder(noise)
        generated_images = generated_images.view(-1, *img_shape)
        elapsed_time = time.time() - start_time
        print(f"Time taken to generate {num_generate:.2f} images: {elapsed_time:.2f} seconds")
    return generated_images

def find_most_similar_images(real_images, generated_images):
    most_similar_images = []
    for real_img in real_images:
        mse_list = [mean_squared_error(real_img.cpu().numpy().flatten(), gen_img.cpu().numpy().flatten()) for gen_img in generated_images]
        mse_tensor = torch.tensor(mse_list)
        best_match = generated_images[torch.argmin(mse_tensor)]
        most_similar_images.append(best_match)
    return torch.stack(most_similar_images)

def compute_radial_intensity(images, img_shape):
    center = (img_shape[1] // 2, img_shape[2] // 2)
    max_distance = int(np.sqrt(center[0]**2 + center[1]**2))
    radial_intensity = np.zeros(max_distance)
    radial_counts = np.zeros(max_distance)
    for img in images:
        img = img.cpu().numpy().squeeze()
        for i in range(img_shape[1]):
            for j in range(img_shape[2]):
                distance = int(np.sqrt((i - center[0])**2 + (j - center[1])**2))
                if distance < max_distance:
                    radial_intensity[distance] += img[i, j]
                    radial_counts[distance] += 1

    radial_intensity /= radial_counts
    return radial_intensity

def calculate_rmae(real, gen):
    return np.mean(np.abs(real - gen) / (np.abs(real) + np.abs(gen) + 1e-10), axis=None)

def two_point_correlation(image, max_distance):
    image = image.squeeze().cpu().numpy()
    height, width = image.shape
    correlation = np.zeros(max_distance)
    counts = np.zeros(max_distance)

    for i in range(height):
        for j in range(width):
            for di in range(-max_distance, max_distance + 1):
                for dj in range(-max_distance, max_distance + 1):
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        distance = int(np.sqrt(di**2 + dj**2))
                        if distance < max_distance:
                            correlation[distance] += image[i, j] * image[ni, nj]
                            counts[distance] += 1

    correlation /= counts
    return correlation

def compute_average_two_point_correlation(images, max_distance):
    correlation_sum = np.zeros(max_distance)
    num_images = len(images)

    for image in images:
        correlation = two_point_correlation(image, max_distance)
        correlation_sum += correlation

    average_correlation = correlation_sum / num_images
    return average_correlation


def calculate_summed_intensities(real_images, generated_images_list, model_names):
    real_intensity_sum = real_images.view(real_images.size(0), -1).sum(dim=1).cpu().numpy()
    all_generated_sums = {encoder: [] for encoder in set(model_names)}

    for i, generated_images in enumerate(generated_images_list):
        encoder = model_names[i]
        generated_intensity_sum = generated_images.view(generated_images.size(0), -1).sum(dim=1).cpu().numpy()
        all_generated_sums[encoder].append(generated_intensity_sum)

    all_intensities = [real_intensity_sum]
    for encoder in all_generated_sums:
        concatenated_sums = np.concatenate(all_generated_sums[encoder], axis=0)
        all_intensities.append(concatenated_sums)

    combined_intensities = np.concatenate(all_intensities)

    return real_intensity_sum, {k: np.concatenate(v, axis=0) for k, v in all_generated_sums.items()}, combined_intensities


def calculate_total_intensities(real_images, generated_images_list, model_names):
    real_intensities = real_images.view(-1).cpu().numpy()
    all_generated_intensities = {encoder: [] for encoder in set(model_names)}

    for i, generated_images in enumerate(generated_images_list):
        encoder = model_names[i]
        generated_intensities = generated_images.view(-1).cpu().numpy()
        all_generated_intensities[encoder].append(generated_intensities)

    all_intensities = [real_intensities]
    for encoder in all_generated_intensities:
        concatenated_intensities = np.concatenate(all_generated_intensities[encoder], axis=0)
        all_intensities.append(concatenated_intensities)

    combined_intensities = np.concatenate(all_intensities)

    return real_intensities, {k: np.concatenate(v, axis=0) for k, v in all_generated_intensities.items()}, combined_intensities


def calculate_peak_intensities(real_images, generated_images_list, model_names):
    real_peak_intensity = real_images.view(real_images.size(0), -1).max(dim=1).values.cpu().numpy()
    all_generated_peaks = {encoder: [] for encoder in set(model_names)}

    for i, generated_images in enumerate(generated_images_list):
        encoder = model_names[i]
        generated_peak_intensity = generated_images.view(generated_images.size(0), -1).max(dim=1).values.cpu().numpy()
        all_generated_peaks[encoder].append(generated_peak_intensity)

    all_intensities = [real_peak_intensity]
    for encoder in all_generated_peaks:
        concatenated_peaks = np.concatenate(all_generated_peaks[encoder], axis=0)
        all_intensities.append(concatenated_peaks)

    combined_intensities = np.concatenate(all_intensities)

    return real_peak_intensity, {k: np.concatenate(v, axis=0) for k, v in all_generated_peaks.items()}, combined_intensities


def calculate_two_point_correlation_score(original_corr, generated_corr):
    return np.mean(np.abs(original_corr - generated_corr) / (original_corr + generated_corr + 1e-10))


def calculate_pca_average_intensity(pca_components):
    return np.mean(np.abs(pca_components), axis=1)

    
def calculate_ssim(real_images, gen_images):
    return np.mean([ssim(real_img.squeeze(), gen_img.squeeze(), data_range=gen_img.max() - gen_img.min()) for real_img, gen_img in zip(real_images.cpu().numpy(), gen_images.cpu().numpy())])


def interpolate_images(model, img_shape, num_steps=20):
    # Generate two random latent vectors
    z1 = torch.randn(1, latent_dim).to(DEVICE)
    z2 = torch.randn(1, latent_dim).to(DEVICE)

    # Interpolation steps
    z_steps = [(1 - alpha) * z1 + alpha * z2 for alpha in np.linspace(0, 1, num_steps)]

    interpolated_images = []

    # Check if the model is conditional
    if hasattr(model, 'decoder') and 'conditional' in str(type(model.decoder)).lower():
        # Generate images from interpolated latent vectors for conditional model
        labels = torch.ones(1, dtype=torch.long).to(DEVICE)  # Assuming label '0'
        labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()  # One-hot encode the labels
        for z in z_steps:
            img = model.decoder(z, labels).view(-1, *img_shape).cpu().detach().numpy()
            interpolated_images.append(img)
    else:
        # Generate images from interpolated latent vectors for non-conditional model
        for z in z_steps:
            img = model.decoder(z).view(-1, *img_shape).cpu().detach().numpy()
            interpolated_images.append(img)

    return interpolated_images


def parse_log_file(file_path):
    log_file_path = file_path.replace("model", "log").replace(".pth", ".txt")
    if not os.path.exists(log_file_path):
        print(f"Log file {log_file_path} does not exist.")
        return 0, 0, 0  # Default values or handle accordingly
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        if not lines:
            print(f"Log file {log_file_path} is empty.")
            return 0, 0, 0  # Default values or handle accordingly

        if "ST" in file_path or 'Dual' in file_path: # If the model is a scattering transform model
            scatter_time_1 = float(lines[0].split(": ")[1].split()[0])
            scatter_time_2 = float(lines[1].split(": ")[1].split()[0])
            train_time_str = lines[2].split(": ")[1]
            train_time = float(''.join(filter(lambda x: x.isdigit() or x == '.', train_time_str.split()[0])))
        else: # If the model is a non-scattering transform model
            scatter_time_1 = 0
            scatter_time_2 = 0
            train_time_str = lines[0].split(": ")[1]
            train_time = float(''.join(filter(lambda x: x.isdigit() or x == '.', train_time_str.split()[0])))
        return scatter_time_1, scatter_time_2, train_time


#######################################################
################# PLOTTING FUNCTIONS ##################
#######################################################


# Define consistent color mapping
colors = {
    'Real': 'black',
    'Alex': 'blue',
    'STMLP': 'orange',
    'lavgSTMLP': 'green',
    'ldiffSTMLP': 'red',
    'Dual': 'purple',
    'CNN': 'cyan'
}
    
def plot_radial_intensity(models_radial_intensity, original_radial_intensity, model_names, title="Radial Intensity", save=True, save_path='./generator/eval/radial.png'):
    print("Plotting radial intensity")

    # Initialize storage for radial intensities by encoder
    all_radial_intensities = {}

    # Collect and group radial intensities by encoder
    for model_name, radial_intensity in zip(model_names, models_radial_intensity):
        encoder = model_name.split('_')[0]  # Use the first part of the model name as the encoder identifier
        if encoder not in all_radial_intensities:
            all_radial_intensities[encoder] = []
        all_radial_intensities[encoder].append(radial_intensity)

    plt.figure(figsize=(10, 6))
    plt.plot(original_radial_intensity, label=f'Real (n={len(test_images)})', color=colors['Real'], linestyle='--')

    # Calculate and plot the mean radial intensity for each encoder
    for encoder, radial_intensities in all_radial_intensities.items():
        mean_radial_intensity = np.mean(radial_intensities, axis=0)
        std_radial_intensity = np.std(radial_intensities, axis=0)  # Calculate standard deviation
        rmae = calculate_rmae(original_radial_intensity, mean_radial_intensity)
        color = colors.get(encoder, 'gray')  # Use a default color if encoder is not in colors dict
        plt.plot(mean_radial_intensity, label=f'{encoder} (n={num_gen}, RMAE={rmae:.3f})', color=color, alpha=0.5)
        plt.errorbar(range(len(mean_radial_intensity)), mean_radial_intensity, yerr=std_radial_intensity, fmt='none', ecolor=color, capsize=2)

    plt.xlabel('Distance from Center')
    plt.ylabel('Average Pixel Intensity')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(save_path) if save else plt.show()
    plt.close()


def plot_stacked_images(real_images, generated_images_list, model_names, save=True, save_path='./generator/eval/stacked_images.png'):
    print("Plotting stacked images")
    
    # Only keep the first occurrence of each model
    seen_models = set()
    unique_indices = []
    for i, name in enumerate(model_names):
        if name not in seen_models:
            seen_models.add(name)
            unique_indices.append(i)
    
    # Update the model names and generated images list
    model_names = [model_names[i] for i in unique_indices]
    generated_images_list = [generated_images_list[i] for i in unique_indices]
    
    num_models = len(generated_images_list)
    
    # Determine the number of images to use based on the smallest set
    num_images_to_use = min(len(real_images), *[len(gen_images) for gen_images in generated_images_list])

    # Select the appropriate number of real and generated images
    real_images = real_images[:num_images_to_use]
    generated_images_list = [gen_images[:num_images_to_use] for gen_images in generated_images_list]

    # Calculate the average pixel intensity for each position
    real_avg_intensity = np.mean([img.cpu().numpy().squeeze() for img in real_images], axis=0)
    generated_avg_intensities = [np.mean([img.cpu().numpy().squeeze() for img in gen_images], axis=0) for gen_images in generated_images_list]

    # Determine the vmin and vmax based on the data
    all_intensities = [real_avg_intensity] + generated_avg_intensities
    vmin = min(np.min(intensity) for intensity in all_intensities)
    vmax = max(np.max(intensity) for intensity in all_intensities)

    fig, axes = plt.subplots(1, num_models + 1, figsize=(15, 5))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)  # Dynamic normalization based on data

    # Plot heatmap for real images
    im = axes[0].imshow(real_avg_intensity, cmap='viridis', norm=norm)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')

    # Plot heatmaps for generated images
    for i, (avg_intensity, name) in enumerate(zip(generated_avg_intensities, model_names)):
        im = axes[i + 1].imshow(avg_intensity, cmap='viridis', norm=norm)
        axes[i + 1].set_title(name, fontsize=12)
        axes[i + 1].axis('off')

    # Adjust the layout to give room for the colorbar
    plt.subplots_adjust(left=0.05, right=0.88, top=0.85, bottom=0.05)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Average Pixel Intensity')
    plt.savefig(save_path) if save else plt.show()
    plt.close()


def plot_intensity_histograms(real_intensity_sum, all_generated_sums, combined_intensities, model_names, xlabel='Intensity sum', save=True, save_path='./generator/eval/summedintensityhist.png'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Histogram for real images
    bins = np.histogram_bin_edges(combined_intensities, bins=30)
    real_hist, _ = np.histogram(real_intensity_sum, bins=bins, density=True)
    real_std = np.sqrt(real_hist / len(real_intensity_sum))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    capsize = 4  # Length of the horizontal caps
    
    ax1.errorbar(bin_centers, real_hist, yerr=real_std, fmt=' ', capsize=capsize, label=f'Real (n={len(test_images)})', color='black', alpha=0.7, elinewidth=1)
    ax1.step(bins, np.append(real_hist, real_hist[-1]), where='post', color=colors['Real']) #np.append(real_hist, real_hist[-1])
    

    # Histogram for generated images
    for encoder in set(model_names):
        concatenated_sums = all_generated_sums[encoder]
        gen_hist, _ = np.histogram(concatenated_sums, bins=bins, density=True)
        gen_std = np.sqrt(gen_hist / len(concatenated_sums))
        rmae = calculate_rmae(real_hist, gen_hist)

        # Fill between needs to start from the edges of the bin, not from the centers
        ax1.step(bins, np.append(gen_hist, gen_hist[-1]), where='post', color=colors[encoder])
        ax1.fill_between(bins[:-1], np.append(gen_hist, gen_hist[-1])[:-1], step='post', color=colors[encoder], alpha=0.3)
        
        ax1.errorbar(bin_centers, gen_hist, yerr=gen_std, fmt=' ', capsize=capsize, 
                    label=f'{encoder} (n={num_gen}, RMAE={rmae:.3f})', color=colors[encoder], alpha=0.7, elinewidth=1)

    ax1.set_yscale('log')
    ax1.set_ylabel('Frequency')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    for encoder in set(model_names):
        concatenated_sums = all_generated_sums[encoder]
        gen_hist_values, _ = np.histogram(concatenated_sums, bins=bins, density=True)
        relative_error = 2 * (real_hist - gen_hist_values) / (real_hist + gen_hist_values + 1e-10)
        ax2.bar(bin_centers, relative_error, width=(bins[1] - bins[0]), alpha=0.4, label=f'{encoder} Relative Error')

    ax2.set_ylabel(r'$ 2 \times \frac{(\mathrm{Real} - \mathrm{Gen})}{\mathrm{Real} + \mathrm{Gen}}$')
    ax2.set_xlabel(xlabel)
    ax2.grid(True)
    ax2.legend()
    
    x_limits = [bin_centers[0] - (bins[1] - bins[0])/2, bin_centers[-1] + (bins[1] - bins[0])/2]
    ax1.set_xlim(x_limits)
    ax2.set_xlim(x_limits)
    ax2.set_ylim(min(-1, np.min(relative_error)), max(1, np.max(relative_error)))

    plt.tight_layout()
    plt.savefig(save_path) if save else plt.show()
    plt.close()

def plot_total_intensity_distribution_sample(real_images, generated_images_list, model_names, num_samples=1, save=True, save_path='./generator/eval/totalintensitydist_sample.png'):
    print("Plotting samples of intensity distribution")
    
    # Ensure the sample size is not greater than the number of available images
    num_samples = min(num_samples, len(real_images))

    # Select a random sample of real images
    real_sample_indices = torch.randperm(len(real_images))[:num_samples]
    real_samples = real_images[real_sample_indices]
    
    # Initialize bins for histogram calculations
    combined_intensities = real_samples.view(-1).cpu().numpy()

    # Variable to keep track of the loop iteration
    loop_count = 0

    for generated_images in generated_images_list:
        loop_count += 1

        if loop_count % 5 == 0:  # Only sample on every fifth iteration
            gen_sample_indices = torch.randperm(len(generated_images))[:num_samples]  # Select num_samples images
            gen_samples = generated_images[gen_sample_indices]
            for gen_sample in gen_samples:
                combined_intensities = np.concatenate((combined_intensities, gen_sample.view(-1).cpu().numpy()))

    bins = np.histogram_bin_edges(combined_intensities, bins=30)  # Shared bins for all histograms
    
    # Different line styles for different samples
    linestyles = ['-', '--', '-.', ':']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot real images' histograms
    for i in range(num_samples):
        real_sample = real_samples[i]
        real_intensities = real_sample.view(-1).cpu().numpy()
        real_hist, _ = np.histogram(real_intensities, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        linestyle = linestyles[i % len(linestyles)]
        ax1.step(bin_centers, real_hist, where='mid', label=f'Real Sample {i + 1}', color=colors['Real'], linestyle=linestyle)
        
        # Plot generated images' histograms (on every fifth iteration)
        for j, (generated_images, model_name) in enumerate(zip(generated_images_list, model_names)):
            if (j + 1) % 5 == 0:  # Process every fifth model
                gen_sample = generated_images[torch.randperm(len(generated_images))[:1]][0]  # Select only one sample per encoder
                gen_intensities = gen_sample.view(-1).cpu().numpy()
                gen_hist, _ = np.histogram(gen_intensities, bins=bins, density=True)
                ax1.step(bin_centers, gen_hist, where='mid', label=f'{model_name} Sample {i + 1}', color=colors[model_name], linestyle=linestyle)
                ax1.fill_between(bin_centers, gen_hist, step='mid', color=colors[model_name], alpha=0.2)

    ax1.set_yscale('log')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Pixel Intensity')
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=num_samples)
    ax1.grid(True)
    ax1.set_title('Total Intensity Distribution')

    # Plot relative error between real and generated histograms (on every fifth iteration)
    for j, (generated_images, model_name) in enumerate(zip(generated_images_list, model_names)):
        if (j + 1) % 5 == 0:  # Process every fifth model
            for i in range(num_samples):
                real_sample = real_samples[i]
                real_intensities = real_sample.view(-1).cpu().numpy()
                real_hist, _ = np.histogram(real_intensities, bins=bins, density=True)
                gen_sample = generated_images[torch.randperm(len(generated_images))[:1]][0]  # Select only one sample per encoder
                gen_intensities = gen_sample.view(-1).cpu().numpy()
                gen_hist, _ = np.histogram(gen_intensities, bins=bins, density=True)

                relative_error = 2 * (real_hist - gen_hist) / (real_hist + gen_hist + 1e-10)
                ax2.bar(bin_centers, relative_error, width=(bins[1] - bins[0]), alpha=0.4, label=f'{model_name} Sample {i + 1} Relative Error')
    
    ax2.set_ylabel(r'$ 2 \times \frac{(\mathrm{Real} - \mathrm{Gen})}{\mathrm{Real} + \mathrm{Gen}}$')
    ax2.set_xlabel('Pixel Intensity')
    ax2.grid(True)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path) if save else plt.show()
    plt.close()


def plot_latent_spaces(latent_representations, model_names, default_perplexity=30, learning_rate=200, n_iter=1000, save_path=None):
    # Only keep the first occurrence of each model
    seen_models = set()
    unique_indices = []
    for i, name in enumerate(model_names):
        if name not in seen_models:
            seen_models.add(name)
            unique_indices.append(i)
    
    # Update the model names and latent representations
    model_names = [model_names[i] for i in unique_indices]
    latent_representations = [latent_representations[i] for i in unique_indices]
    
    num_models = len(latent_representations)
    
    if num_models == 1:
        fig, axes = plt.subplots(1, num_models, figsize=(15, 5))
        axes = [axes]  # Convert single Axes object to a list
    else:
        fig, axes = plt.subplots(1, num_models, figsize=(15, 5))

    sc = None  # Initialize `sc` to None

    for i, (latents, name) in enumerate(zip(latent_representations, model_names)):
        # Check if latents is a valid array with more than one dimension
        if latents is None or isinstance(latents, (float, np.float32, np.float64)) or not hasattr(latents, 'shape') or len(latents.shape) < 2:
            print(f"Skipping {name} due to invalid latent representation: {latents.shape if latents is not None and hasattr(latents, 'shape') else 'None'}")
            continue

        # Ensure latents is a 2D array
        if len(latents.shape) == 1:
            latents = latents.reshape(1, -1)
        elif len(latents.shape) > 2:
            latents = latents.reshape(latents.shape[0], -1)

        # Adjust perplexity to avoid issues
        perplexity = min(default_perplexity, latents.shape[0] - 1)
        
        tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
        tsne_results = tsne.fit_transform(latents)
        
        sc = axes[i].scatter(tsne_results[:, 0], tsne_results[:, 1], c=latents[:, 0], cmap='viridis', alpha=0.6)
        axes[i].set_title(name)
        axes[i].axis('off')

    # Only add a colorbar if `sc` has been assigned (i.e., at least one plot was made)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Latent Space Value')

    plt.savefig(save_path, dpi=300, bbox_inches='tight') if save_path else plt.show()



def plot_interpolated_images(interpolated_images, save=True, save_path='./interpolation_graph.png'):
    num_steps = len(interpolated_images)
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
    for i, img in enumerate(interpolated_images):
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path) if save else plt.show()
    plt.close()
    

def calculate_statistics(time_data):
    means = time_data.mean(axis=0)
    stds = time_data.std(axis=0)
    return means, stds

def plot_generation_time(models, save=True, save_path='./generator/eval/timehist.png'):
    scatter_times_1 = {}
    scatter_times_2 = {}
    train_times = {}
    model_names = []

    for model_info in models:
        if os.path.exists(model_info["path"]):
            scatter_time_1, scatter_time_2, train_time = parse_log_file(model_info["path"])
            model_name = model_info["name"]
            
            if model_name not in scatter_times_1:
                scatter_times_1[model_name] = []
                scatter_times_2[model_name] = []
                train_times[model_name] = []
            
            scatter_times_1[model_name].append(scatter_time_1)
            scatter_times_2[model_name].append(scatter_time_2)
            train_times[model_name].append(train_time)
            if model_name not in model_names:
                model_names.append(model_name)

    # Convert the lists into numpy arrays to calculate statistics
    means_dict = {}
    stds_dict = {}
    
    for model_name in model_names:
        times_array = np.array([scatter_times_1[model_name], scatter_times_2[model_name], train_times[model_name]]).T
        means, stds = calculate_statistics(times_array)
        means_dict[model_name] = means
        stds_dict[model_name] = stds
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    index = np.arange(len(model_names))

    # Create bars for each model with standard deviation as error bars
    for i, model_name in enumerate(model_names):
        means = means_dict[model_name]
        stds = stds_dict[model_name]
        total_means = np.sum(means)
        total_stds = np.sqrt(np.sum(stds**2))
        
        # Scattering times go first
        plt.bar(index[i], means[0], yerr=stds[0], color='blue', alpha=0.4, capsize=4, error_kw={'ecolor': 'blue'})
        plt.bar(index[i], means[1], yerr=stds[1], color='orange', alpha=0.4, bottom=means[0], capsize=4, error_kw={'ecolor': 'orange'})
        # Training time goes on top
        plt.bar(index[i], means[2], yerr=stds[2], color='green', alpha=0.4, bottom=means[0]+means[1], capsize=4, error_kw={'ecolor': 'green'})
        
        # Error bar for total time
        plt.errorbar(index[i], total_means, yerr=total_stds, fmt='', color='black', alpha=0.4, capsize=10, markeredgewidth=2)

    plt.xticks(index, model_names)
    plt.ylabel('Time (seconds)')
    plt.title('Total Computation and Training Time')

    # Custom legend for colors only
    custom_legend = [
        plt.Line2D([0], [0], color='black', marker='', linestyle='None', markersize=10, label='Total Time Error'),
        plt.Line2D([0], [0], color='green', lw=4, label='Training Time'),
        plt.Line2D([0], [0], color='orange', lw=4, label='Scattering Time (Test Data)'),
        plt.Line2D([0], [0], color='blue', lw=4, label='Scattering Time (Training Data)')
    ]

    plt.legend(handles=custom_legend, loc='best')  # 'best' places the legend in the optimal location
    plt.grid(True)

    plt.savefig(save_path) if save else plt.show()



###################################################################################
########################### MAIN SCRIPT ###########################################
###################################################################################

real_features = extract_features(test_images)

# Initialize empty lists to store metrics for each fold
summary_table, reconstructed_images_list, generated_images_list, latent_representations, model_names, log_files, models = [], [], [], [], [], [], []

for encoder in encoder_list:
    # Initialize lists to store metrics for all folds for the current model
    all_mse, all_kl_divergence, all_silhouette, all_fid = [], [], [], []

    for fold in range(5) if FFCV else [6]:
        path = get_model_name(galaxy_classes, num_galaxies, encoder, fold)
        
        if os.path.exists(path):
            if "lavgSTMLP" in encoder:
                scatshape = lavg_scatshape
            elif "ldiffSTMLP" in encoder:
                scatshape = ldiff_scatshape
            else:
                scatshape = np.shape(scat_coeffs)

            model = load_model(path, scatshape=scatshape, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, num_classes=num_classes)
            models.append({"name": encoder, "path": path})
            reconstructed_images, latent_rep, mse, kl_divergence, silhouette = reconstruct_and_evaluate(model, test_images, scat_coeffs, lavg_scat_coeffs, ldiff_scat_coeffs, test_labels, path)
            generated_images = generate_from_noise(model, latent_dim, num_generate) 
            gen_features = extract_features(generated_images)
            fid = calculate_fid(real_features, gen_features)
            
            # Append the metrics for this fold to the lists
            all_mse.append(mse)
            all_kl_divergence.append(kl_divergence)
            all_silhouette.append(silhouette)
            all_fid.append(fid)
            
            # Append the reconstructed and generated images, as well as latent representations
            reconstructed_images_list.append(reconstructed_images)
            generated_images_list.append(generated_images)
            latent_representations.append(latent_rep)
            model_names.append(encoder)

    # Only calculate and store the summary if there are valid results for the model
    if all_mse:  # This checks if any metrics were appended, meaning a valid path existed
        mse_mean = np.mean(all_mse)
        mse_std = np.std(all_mse)
        kl_divergence_mean = np.mean(all_kl_divergence)
        kl_divergence_std = np.std(all_kl_divergence)
        silhouette_mean = np.mean(all_silhouette)
        silhouette_std = np.std(all_silhouette)
        fid_mean = np.mean(all_fid)
        fid_std = np.std(all_fid)

        # Store the summary for the current model
        summary_table.append({
            "Model": encoder,
            "MSE Mean": mse_mean,
            "MSE Std": mse_std,
            "KL Divergence Mean": kl_divergence_mean,
            "KL Divergence Std": kl_divergence_std,
            "Silhouette Score Mean": silhouette_mean,
            "Silhouette Score Std": silhouette_std,
            "FID Mean": fid_mean,
            "FID Std": fid_std
        })

# After processing all encoders, you can convert the summary table to a DataFrame, etc.

# Convert to DataFrame
summary_df = pd.DataFrame(summary_table)
summary_df = summary_df.fillna(0)  # Fill NaN values with 0

# Save or print the summary
print("Summary DataFrame:\n", summary_df)
summary_df.to_csv(f"{save_dir}/summary_df.csv", index=False)

# If you want to identify the best model based on a composite score:
summary_df["Overall Score"] = (
    0.25 * (1 - summary_df["MSE Mean"] / summary_df["MSE Mean"].max()) +
    0.25 * (1 - summary_df["KL Divergence Mean"] / summary_df["KL Divergence Mean"].max()) +
    0.5 * (summary_df["Silhouette Score Mean"] / summary_df["Silhouette Score Mean"].max())
)

# Find and print the best model
best_model = summary_df.loc[summary_df["Overall Score"].idxmax()]
print("Best Model:", best_model["Model"])
with open(f"./generator/eval/{galaxy_classes}_log.txt", "w") as file:
    file.write(summary_df.to_string())
    file.write("\nBest Model: " + best_model["Model"])

#Create latex table
summary_df.to_latex(f"{save_dir}/summary_df.tex", index=False)


# RECONSTRUCTED AND GENERATED IMAGES
vae_multmod(test_images[:num_display], reconstructed_images_list, model_names, f"Reconstruction of {galaxy_classes}", 
            save=True, save_path=f"{save_dir}/reconstructed.png")
vae_multmod(test_images[:num_display], generated_images_list, model_names, f"Generated {galaxy_classes}",
            save=True, save_path=f"{save_dir}/generated.png")

# SIMILAR GENERATED IMAGES AND ORIGINAL IMAGES
most_similar_images_list = []
for generated_images in generated_images_list:
    most_similar_images = find_most_similar_images(test_images[:num_display], generated_images)
    most_similar_images_list.append(most_similar_images)
vae_multmod(test_images[:num_display], most_similar_images_list, model_names, f"Generated images most similar to original {galaxy_classes}",
            save=True, save_path=f"{save_dir}/generated_similar.png")


# PIXEL DISTRIBUTIONS
# Calculate summed intensities and plot
real_intensity_sum, all_generated_sums, combined_intensities = calculate_summed_intensities(test_images, generated_images_list, model_names)
plot_intensity_histograms(real_intensity_sum, all_generated_sums, combined_intensities, model_names, xlabel='Intensity sum', save_path=f"{save_dir}/summedintensitydist.png")

# Calculate peak intensities and plot
real_peak_intensity, all_generated_peaks, combined_peaks = calculate_peak_intensities(test_images, generated_images_list, model_names)
plot_intensity_histograms(real_peak_intensity, all_generated_peaks, combined_peaks, model_names, xlabel='Peak Intensity', save_path=f"{save_dir}/peakintensitydist.png")

# Calculate total intensities and plot
real_total_intensity, all_generated_total_intensities, combined_total_intensities = calculate_total_intensities(test_images, generated_images_list, model_names)
plot_intensity_histograms(real_total_intensity, all_generated_total_intensities, combined_total_intensities, model_names, xlabel='Total Intensity', save_path=f"{save_dir}/totalintensitydist.png")

print("Models: ", models)
plot_total_intensity_distribution_sample(test_images, generated_images_list, model_names, save_path=f"{save_dir}/totalintensitydist_sample.png")

# RADIAL INTENSITY COMPARISON
original_radial_intensity = compute_radial_intensity(test_images[:num_display], img_shape)
models_radial_intensity = []
num_images_per_model = {"original": len(test_images)}
for i, generated_images in enumerate(generated_images_list):
    radial_intensity = compute_radial_intensity(generated_images, img_shape)
    models_radial_intensity.append(radial_intensity)
    num_images_per_model[model_names[i]] = len(generated_images)
title = f'Radial Intensity of {galaxy_classes}'
plot_radial_intensity(models_radial_intensity, original_radial_intensity, model_names, title, save_path=f"{save_dir}/radial.png")
plot_stacked_images(test_images, generated_images_list, model_names, save_path=f"{save_dir}/stacked_images.png")

plot_generation_time(models, save_path=f'{save_dir}/timehist.png')

plot_latent_spaces(latent_representations, model_names, save_path=f'{save_dir}/latent_spaces.png')






"""


def plot_two_point_correlation(original_corr, generated_corr_list, model_names, max_distance, save=True, save_path='./generator/eval/two_point_correlation.png'):
    print("Plotting two-point correlation function")
    distances = np.arange(max_distance)
    plt.figure(figsize=(10, 6))
    plt.plot(distances, original_corr, label='Original', color='black', linestyle='--')
    for i, generated_corr in enumerate(generated_corr_list):
        plt.plot(distances, generated_corr, label=model_names[i])
    plt.xlabel('Distance')
    plt.ylabel('Two-Point Correlation')
    plt.title('Two-Point Correlation Function')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(save_path) if save else plt.show()
    plt.close()

# Plot histogram of PCA component average intensities for different models
def plot_pca_average_intensity_histogram(average_intensities, model_names, save=True, save_path='./generator/eval/pcahistogram.png'):
    n_components = average_intensities[0].shape[0]
    x = np.arange(n_components)
    bar_width = 0.1  # Width of the bars

    plt.figure(figsize=(14, 8))

    for i, avg_intensity in enumerate(average_intensities):
        plt.bar(x + i * bar_width, avg_intensity, width=bar_width, label=model_names[i])

    plt.xlabel('PCA Component')
    plt.ylabel('Average Intensity')
    plt.title('Average Intensity of PCA Components for Different Models')
    plt.xticks(x + bar_width * (len(model_names) - 1) / 2, [f'PC{i+1}' for i in x])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(save_path) if save else plt.show()
    plt.close()


def calculate_scattering_intensity(images, scattering):
    scat_coeffs = scattering(images.contiguous().to(DEVICE))
    scat_coeffs = scat_coeffs.view(scat_coeffs.size(0), -1)  # Flatten
    return scat_coeffs.cpu().numpy()

def perform_pca(scattering_coeffs, n_components=10):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scattering_coeffs)
    return pca_result, pca.components_


def plot_scattering_intensity(original_avg_intensity, generated_avg_intensity_list, model_names, save=True, save_path='./generator/eval/scatteringspectrum.png'):
    print("Plotting scattering spectrum")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    num_coeffs = len(original_avg_intensity)
    x = np.arange(num_coeffs)

    # Calculate RMAE for each model
    rmae_scores = []
    for gen_avg_intensity in generated_avg_intensity_list:
        gen_avg_intensity_np = gen_avg_intensity.cpu().numpy()  # Convert to numpy array
        original_avg_intensity_np = original_avg_intensity.cpu().numpy()  # Convert to numpy array
        rmae = calculate_rmae(original_avg_intensity_np, gen_avg_intensity_np)
        rmae_scores.append(rmae)

    # Plot the average scattering intensities
    ax1.plot(x, original_avg_intensity, label='Original (n={})'.format(len(test_images)), color='black', linestyle='--')
    for i, gen_avg_intensity in enumerate(generated_avg_intensity_list):
        ax1.plot(x, gen_avg_intensity, label='{} (n={num_generate}, RMAE={:.3f})'.format(model_names[i], rmae_scores[i]))
    
    ax1.set_xlabel('Scattering Coefficient Channel')
    ax1.set_ylabel('Average Intensity (log scale)')
    ax1.set_yscale('log')
    ax1.set_title('Scattering Coefficients Average Intensity')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot relative error for all models
    for i, gen_avg_intensity in enumerate(generated_avg_intensity_list):
        gen_avg_intensity_np = gen_avg_intensity.cpu().numpy()  # Convert to numpy array
        original_avg_intensity_np = original_avg_intensity.cpu().numpy()  # Convert to numpy array
        relative_error = 2 * (original_avg_intensity_np - gen_avg_intensity_np) / (original_avg_intensity_np + gen_avg_intensity_np + 1e-10)
        ax2.bar(x, relative_error, width=0.8, alpha=0.4, label='{} Relative Error'.format(model_names[i]))

    ax2.set_ylabel(r'$ 2 \times \frac{(\mathrm{Real} - \mathrm{Gen})}{\mathrm{Real} + \mathrm{Gen}}$')
    ax2.set_xlabel('Scattering Coefficient Channel')
    ax2.grid(True)
    ax2.legend(loc='upper right')
    plt.savefig(save_path) if save else plt.show()
    plt.close()
    

def plot_pca_scattering_coeffs(test_images, generated_images_list, scattering, model_names, save=True, save_path='./generator/eval/pcascattering.png'):
    print("Plotting PCA of scattering coefficients")
    pca = PCA(n_components=3)

    # Calculate scattering intensity for original images
    original_scat_intensity = calculate_scattering_intensity(test_images, scattering)
    generated_scat_intensity_list = [calculate_scattering_intensity(generated_images, scattering) for generated_images in generated_images_list]

    # Combine all intensities and fit PCA
    all_intensities = np.vstack([original_scat_intensity] + generated_scat_intensity_list)
    pca_result = pca.fit_transform(all_intensities)

    # Separate PCA results for original and generated images
    original_pca_result = pca_result[:len(test_images)]
    generated_pca_results = []
    start_idx = len(test_images)
    for generated_scat_intensity in generated_scat_intensity_list:
        end_idx = start_idx + len(generated_scat_intensity)
        generated_pca_results.append(pca_result[start_idx:end_idx])
        start_idx = end_idx

    # Plot the PCA results
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original images
    ax.scatter(original_pca_result[:, 0], original_pca_result[:, 1], original_pca_result[:, 2], label=f'Original (n={len(test_images)})', alpha=0.05, color='gray', marker='o')

    # Plot generated images for each model
    colors = plt.get_cmap('tab10')(range(len(model_names)))
    for idx, (gen_pca_result, model_name) in enumerate(zip(generated_pca_results, model_names)):
        ax.scatter(gen_pca_result[:, 0], gen_pca_result[:, 1], gen_pca_result[:, 2], label=f'{model_name} (n={num_generate})', alpha=0.05, color=colors[idx], marker='o')

    # Plot means after scatter plots to ensure stars appear on top
    ax.scatter(original_pca_result[:, 0].mean(), original_pca_result[:, 1].mean(), original_pca_result[:, 2].mean(), label='Original Mean', color='black', marker='*', s=300)
    for idx, (gen_pca_result, model_name) in enumerate(zip(generated_pca_results, model_names)):
        ax.scatter(gen_pca_result[:, 0].mean(), gen_pca_result[:, 1].mean(), gen_pca_result[:, 2].mean(), label=f'{model_name} Mean', color=colors[idx], marker='*', s=300)

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.title('PCA of Scattering Coefficients')
    plt.legend(loc='upper right')
    plt.savefig(save_path) if save else plt.show()
    plt.close()


plot_two_point_correlation(original_two_point_corr, generated_two_point_corr_list, model_names, max_distance, save_path=f"{save_dir}/two_point_correlation.png")

original_avg_intensity = scattering(test_images.contiguous().to(DEVICE)).view(len(test_images), -1).mean(dim=0).cpu()
generated_avg_intensity_list = []
for generated_images in generated_images_list:
    gen_scat_coeffs = scattering(generated_images.contiguous().to(DEVICE)).view(len(generated_images), -1).mean(dim=0).cpu()
    generated_avg_intensity_list.append(gen_scat_coeffs)
plot_scattering_intensity(original_avg_intensity, generated_avg_intensity_list, model_names, save_path=f"{save_dir}/scatteringspectrum.png")
plot_pca_scattering_coeffs(test_images, generated_images_list, scattering, model_names, save_path=f"{save_dir}/pcascattering.png")"""
