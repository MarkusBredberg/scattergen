import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from utils.plotting import vae_multmod
import torch
from utils.calc_tools import get_main_and_secondary_peaks, get_main_and_secondary_peaks_with_locations
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression 

print("Running 3.2")

# Configuration for auto-loading models
galaxy_classes, num_galaxies = 11, 1008
hidden_dim1, hidden_dim2, latent_dim = 256, 128, 64
J, L, order = 2, 12, 2
num_display, num_generate = 5, 1000 # Plotted images per model, generated images for statistics

include_two_point_correlation = False
FFCV = True # Use five-fold cross-validation

num_gen = num_generate * 5 if FFCV else num_generate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(galaxy_classes) if isinstance(galaxy_classes, list) else 1

# Load the saved data
save_dir = f'./generator/eval/{galaxy_classes}_last_evaluation'

loaded_data = torch.load(os.path.join(save_dir, f'compressed_{galaxy_classes}_{num_galaxies}_data.pt'))
scat_coeffs = loaded_data['scat_coeffs'].float()  # Convert back to float32 if needed
lavg_scat_coeffs = loaded_data['lavg_scat_coeffs'].float()
ldiff_scat_coeffs = loaded_data['ldiff_scat_coeffs'].float()
test_images = loaded_data['test_images'].float()
test_labels = loaded_data['test_labels']
reconstructed_images_list = [img.float() for img in loaded_data['reconstructed_images_list']]
generated_images_list = [img.float() for img in loaded_data['generated_images_list']]
latent_representations = [rep.float() for rep in loaded_data['latent_representations']]
model_names = loaded_data['model_names']
models = loaded_data['models']

print("Data successfully loaded from the PyTorch compressed file.")



img_shape = test_images[0].shape

print("Model names", model_names)
print("Models", models)

if generated_images_list is not None and len(generated_images_list) > 0:
    print(f"Generated {len(generated_images_list[0])*len(generated_images_list)} images.")
else:
    print("No images were generated.")



###########################################################
################# EVALUATION FUNCTIONS #####################
###########################################################


def find_most_similar_images(real_images, generated_images):
    most_similar_images = []
    for real_img in real_images:
        real_img_flat = real_img.flatten().to(generated_images.device)
        mse_list = [(real_img_flat - gen_img.flatten()).pow(2).mean() for gen_img in generated_images]
        mse_tensor = torch.tensor(mse_list, device=generated_images.device)
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


def calculate_rmae(real, gen):  # Relative mean absolute error
    return np.mean(np.abs(real - gen)) / (np.mean(real) + 1e-8)


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
        return 0, 0, 0, 0  # Default values or handle accordingly

    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) < 3:
            print(f"Log file {log_file_path} does not have enough lines.")
            return 0, 0, 0, 0  # Default values or handle accordingly

        try:
            if "ST" in file_path or 'Dual' in file_path:  # If the model is a scattering transform model
                scatter_time_1 = float(lines[0].split(": ")[1].split()[0])
                scatter_time_2 = float(lines[1].split(": ")[1].split()[0])
                train_time_str = lines[2].split(": ")[1].strip()
                train_time = float(train_time_str.split()[0])
                epochs_str = lines[3].split(": ")[1].strip()
                epochs = float(epochs_str.split()[0]) 
            else:  # If the model is a non-scattering transform model
                scatter_time_1 = 0
                scatter_time_2 = 0
                train_time_str = lines[0].split(": ")[1]
                train_time = float(''.join(filter(lambda x: x.isdigit() or x == '.', train_time_str.split()[0])))
                epochs = float(lines[1].split(": ")[1])
        except (IndexError, ValueError) as e:
            print(f"Error parsing log file {log_file_path}: {e}")
            return 0, 0, 0, 0  # Default values or handle accordingly
                
        return scatter_time_1, scatter_time_2, train_time, epochs
    

def old_evaluate_symmetry(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Convert to NumPy if it's a tensor
    
    gray_image = np.squeeze(image, axis=0) #Reduce grayscale image dimension
    height, width = gray_image.shape
    
    # Vertical symmetry (left vs. right)
    left_half = gray_image[:, :width // 2]
    right_half = gray_image[:, width // 2:]
    right_half_flipped = np.flip(right_half, axis=1)
    if left_half.shape[1] != right_half_flipped.shape[1]:
        right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))
    vertical_diff = np.sum(np.abs(left_half - right_half_flipped))
    
    # Horizontal symmetry (top vs. bottom)
    top_half = gray_image[:height // 2, :]
    bottom_half = gray_image[height // 2:, :]
    bottom_half_flipped = np.flip(bottom_half, axis=0)
    if top_half.shape[0] != bottom_half_flipped.shape[0]:
        bottom_half_flipped = cv2.resize(bottom_half_flipped, (top_half.shape[1], top_half.shape[0]))
    horizontal_diff = np.sum(np.abs(top_half - bottom_half_flipped))
    
    # Overall asymmetry score is the average of vertical and horizontal scores
    vertical_score = vertical_diff / (width * height)
    horizontal_score = horizontal_diff / (width * height)
    overall_asymmetry_score = (vertical_score + horizontal_score) / 2
    
    return vertical_score, horizontal_score, overall_asymmetry_score

def evaluate_symmetry(image):
    """
    Calculate a symmetry score by folding the image twice (first vertically, then horizontally),
    and comparing how well the regions overlap.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Convert to NumPy if it's a tensor
    
    gray_image = np.squeeze(image, axis=0)  # Reduce grayscale image dimension
    height, width = gray_image.shape
    
    # Vertical fold: Left half folded over the right half
    left_half = gray_image[:, :width // 2]
    right_half = gray_image[:, width // 2:]
    right_half_flipped = np.flip(right_half, axis=1)
    
    if left_half.shape[1] != right_half_flipped.shape[1]:
        right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))
    
    # Combine the left and right halves (after folding) to get a vertically folded image
    vertical_fold = (left_half + right_half_flipped) / 2
    
    # Horizontal fold: Top half folded over the bottom half of the vertically folded image
    top_half = vertical_fold[:height // 2, :]
    bottom_half = vertical_fold[height // 2:, :]
    bottom_half_flipped = np.flip(bottom_half, axis=0)
    
    if top_half.shape[0] != bottom_half_flipped.shape[0]:
        bottom_half_flipped = cv2.resize(bottom_half_flipped, (top_half.shape[1], top_half.shape[0]))
    
    # Calculate symmetry difference after both folds
    double_fold_diff = np.sum(np.abs(top_half - bottom_half_flipped))
    
    # Normalize by the image size
    double_fold_score = double_fold_diff / (width * height)
    
    return double_fold_score


def calculate_mse(original, reconstructed):
    # Convert PyTorch tensor to NumPy array and ensure it's in the correct format
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()  # Convert to NumPy if it's a tensor
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()

    # Ensure grayscale images are properly formatted
    if len(original.shape) == 3 and original.shape[0] == 1:
        original = np.squeeze(original, axis=0)
    if len(reconstructed.shape) == 3 and reconstructed.shape[0] == 1:
        reconstructed = np.squeeze(reconstructed, axis=0)
    
    # Convert to grayscale if necessary
    if len(original.shape) == 3:  # If the image has color channels
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
    
    if len(reconstructed.shape) == 3:  # If the image has color channels
        reconstructed_gray = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    else:
        reconstructed_gray = reconstructed
    
    # Resize reconstructed image if necessary
    if original_gray.shape != reconstructed_gray.shape:
        reconstructed_gray = cv2.resize(reconstructed_gray, (original_gray.shape[1], original_gray.shape[0]))
    
    # Calculate Mean Squared Error
    mse_value = np.mean((original_gray - reconstructed_gray) ** 2)
    return mse_value



#######################################################
################# PLOTTING FUNCTIONS ##################
#######################################################


# Define consistent color mapping
colors = {
    'Real': 'black',
    'Alex': 'slategray',
    'STMLP': 'darkorange',
    'lavgSTMLP': 'purple',
    'ldiffSTMLP': 'saddlebrown',
    'Dual': 'limegreen',
    'CNN': 'cornflowerblue'
}


def plot_asymmetry_vs_mse(original_images, reconstructed_images, encoder, save_path='./generator/eval/asymmetry_vs_mse_with_regression.png'):
    asymmetry_scores = []
    mse_values = []
    
    for original, reconstructed in zip(original_images, reconstructed_images):
        # Calculate asymmetry score for the original image
        asymmetry_score = evaluate_symmetry(original)
        asymmetry_scores.append(asymmetry_score)
        
        # Calculate MSE between original and reconstructed image
        mse_value = calculate_mse(original, reconstructed)
        mse_values.append(mse_value)
    
    # Convert lists to numpy arrays for regression
    asymmetry_scores = np.array(asymmetry_scores).reshape(-1, 1)
    mse_values = np.array(mse_values)
    
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(asymmetry_scores, mse_values)
    
    # Predict MSE values using the linear regression model
    predicted_mse_values = model.predict(asymmetry_scores)
    
    # Create a scatter plot and plot the linear regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(asymmetry_scores, mse_values, color='blue', label='Asymmetry vs MSE')
    plt.plot(asymmetry_scores, predicted_mse_values, color='red', label='Linear Regression Fit')
    plt.xlabel('Asymmetry Score')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'Asymmetry Score vs MSE for {encoder}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    

def plot_asymmetry_score_histogram(images, num_bins=20, save_path='./generator/eval/asymmetry_histogram.png'):
    """
    Calculate asymmetry scores for a set of images and plot a histogram showing the distribution of the scores.

    Parameters:
    - images: list or tensor of images to calculate the asymmetry scores for.
    - num_bins: Number of bins for the histogram (default is 20).
    """
    # Step 1: Calculate asymmetry scores for each image
    asymmetry_scores = [evaluate_symmetry(image) for image in images]
    
    # Step 2: Extract the overall asymmetry score
    overall_asymmetry_scores = asymmetry_scores

    # Step 3: Plot histogram of asymmetry scores
    plt.figure(figsize=(8, 6))
    plt.hist(overall_asymmetry_scores, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Asymmetry Score')
    plt.ylabel('Number of Images')
    plt.title('Histogram of Asymmetry Scores')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
def plot_images_by_asymmetry_bins(images, reconstructed_images, save_path='./generator/eval/asymmetry_bins.png'):
    # Step 1: Calculate asymmetry scores for each image
    asymmetry_scores = []
    
    for original, reconstructed in zip(images, reconstructed_images):
        asymmetry_score = evaluate_symmetry(original)
        asymmetry_scores.append(asymmetry_score)
    
    # Step 2: Sort the asymmetry scores and corresponding images
    asymmetry_scores = np.array(asymmetry_scores)
    sorted_indices = np.argsort(asymmetry_scores)
    sorted_scores = asymmetry_scores[sorted_indices]
    sorted_images = [images[i] for i in sorted_indices]
    sorted_reconstructed = [reconstructed_images[i] for i in sorted_indices]
    
    # Step 3: Divide into nine bins and select images from first, third, fifth, seventh, and ninth bins
    num_images = len(sorted_images)
    bin_size = num_images // 9
    
    # Ensure bin indices are within bounds
    first_bin_idx = bin_size - 1 if bin_size > 0 else 0
    third_bin_idx = 3 * bin_size - 1
    fifth_bin_idx = 5 * bin_size - 1
    seventh_bin_idx = 7 * bin_size - 1
    ninth_bin_idx = 9 * bin_size - 1
    
    # Select images and reconstructed images from the specified bins
    selected_bins = [first_bin_idx, third_bin_idx, fifth_bin_idx, seventh_bin_idx, ninth_bin_idx]
    
    selected_images = [sorted_images[i] for i in selected_bins]
    selected_reconstructed_images = [sorted_reconstructed[i] for i in selected_bins]
    selected_scores = [sorted_scores[i] for i in selected_bins]
    
    # Step 4: Plot the images in five columns: one for each bin (first, third, fifth, seventh, ninth)
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    
    # Row 1: Original images
    for i, (image, score) in enumerate(zip(selected_images, selected_scores)):
        axs[0, i].imshow(image.squeeze(), cmap='gray')
        axs[0, i].set_title(f'Asymmetry Score: {score:.4f} (Bin {i*2 + 1})')
        axs[0, i].axis('off')
    
    # Row 2: Reconstructed images
    for i, reconstructed_image in enumerate(selected_reconstructed_images):
        axs[1, i].imshow(reconstructed_image.squeeze(), cmap='gray')
        axs[1, i].set_title(f'Reconstructed Image (Bin {i*2 + 1})')
        axs[1, i].axis('off')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'Plot saved at {save_path}')


    

def plot_images_with_peaks(images, peaks_info_all, save_path='./generator/eval/peaks.png'):
    # Ensure we only plot the first 3 images and their peaks
    images = images[:3]
    peaks_info_all = peaks_info_all[:3]
    
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))  # Set up a 1x3 grid for images
    
    for i, (image, peaks) in enumerate(zip(images, peaks_info_all)):
        axs[i].imshow(image.squeeze(), cmap='viridis')  

        main_peak_location = peaks.get('main_peak_location')
        second_peak_location = peaks.get('second_peak_location')

        if main_peak_location is not None and len(main_peak_location) == 2:
            axs[i].scatter(main_peak_location[1], main_peak_location[0], facecolor='none', edgecolor='gold', s=100, marker='o')

        if second_peak_location is not None and len(second_peak_location) == 2:
            axs[i].scatter(second_peak_location[1], second_peak_location[0], facecolor='none', edgecolor='silver', s=100, marker='o')

        axs[i].set_title(f"Image {i+1}")
        axs[i].axis('off')   
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def plot_peak_statistics(peak_intensity_stats_real, peak_intensity_stats_produced, model_names, title, save_path, show_secondary=True):
    categories_produced = ['Real'] + model_names

    # Convert PyTorch tensors to NumPy arrays if they are not already
    if isinstance(peak_intensity_stats_real, torch.Tensor):
        peak_intensity_stats_real = peak_intensity_stats_real.cpu().numpy()
    if isinstance(peak_intensity_stats_produced, torch.Tensor):
        peak_intensity_stats_produced = [prod.cpu().numpy() for prod in peak_intensity_stats_produced]

    # Split peak intensity stats for real data
    print("Peak_intensity_stats_real:", peak_intensity_stats_real)
    print("Shape of the above:", np.shape(peak_intensity_real))
    real_primary_peaks = np.array(peak_intensity_stats_real['main_peak_value'])
    real_secondary_peaks = np.array(peak_intensity_stats_real['second_peak_value'])

    
    # Calculate mean and std for real peaks
    real_primary_mean, real_primary_std = np.mean(real_primary_peaks), np.std(real_primary_peaks)
    real_secondary_mean, real_secondary_std = np.mean(real_secondary_peaks), np.std(real_secondary_peaks)

    prod_primary_means, prod_primary_stds, prod_secondary_means, prod_secondary_stds = [], [], [], []
    for i in range(len(peak_intensity_stats_produced)):  # Loop over all the different encoders
        # Split for reconstructed data (each encoder is a list)
        prod_primary_peaks = np.array(peak_intensity_stats_produced['main_peak_value'][i]).flatten()
        prod_secondary_peaks = np.array(peak_intensity_stats_produced['second_peak_value'][i]).flatten()


        # Calculate mean and std for reconstructed peaks
        prod_primary_means.append(np.mean(prod_primary_peaks))
        prod_primary_stds.append(np.std(prod_primary_peaks))
        prod_secondary_means.append(np.mean(prod_secondary_peaks))
        prod_secondary_stds.append(np.std(prod_secondary_peaks))
        

    # Plotting function with option to print values on bars
    def plot_with_values(ax, index, primary_means, primary_stds, secondary_means, secondary_stds, categories, title, show_secondary):
        bar_width = 0.35

        # Plot primary peak data
        colors_used = [colors.get(cat, 'gray') for cat in categories]  # Use your defined color mapping
        bars_primary = ax.bar(index, primary_means, bar_width, yerr=primary_stds, label='Primary Peak', color=colors_used, capsize=5)
        
        # Plot secondary peak data only if show_secondary is True
        if show_secondary:
            bars_secondary = ax.bar(index + bar_width, secondary_means, bar_width, yerr=secondary_stds, label='Secondary Peak', color=colors_used, alpha=0.7, capsize=5)

        # Add text labels above the bars
        for bar in bars_primary:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

        if show_secondary:
            for bar in bars_secondary:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

        # Customize plot labels
        ax.set_xlabel('Model')
        ax.set_ylabel('Peak Intensity')
        ax.set_title(title)
        ax.set_xticks(index + (bar_width / 2) if show_secondary else bar_width / 2)
        ax.set_xticklabels(categories)
        ax.legend()

    # Plot produced vs real
    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(len(categories_produced))

    real_primary_mean = [real_primary_mean]  # Make real primary mean a list
    real_primary_std = [real_primary_std]    # Make real primary std a list
    real_secondary_mean = [real_secondary_mean]
    real_secondary_std = [real_secondary_std]

    plot_with_values(ax, index, real_primary_mean + prod_primary_means, real_primary_std + prod_primary_stds,
                     real_secondary_mean + prod_secondary_means, real_secondary_std + prod_secondary_stds,
                     categories_produced, title, show_secondary)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_peak_ratio_distribution(peak_intensity_stats_real, peak_intensity_stats_produced, model_names, save_path):
    # Calculate ratios
    real_ratios = np.array(peak_intensity_stats_real['main_peak_value']) / np.clip(peak_intensity_stats_real['second_peak_value'], 1e-8, None)
    prod_ratios = [prod_stats['second_peak_value'] / np.clip(prod_stats['main_peak_value'], 1e-8, None) for prod_stats in peak_intensity_stats_produced]
    
    # Combine data for plotting
    data = [real_ratios] + prod_ratios
    labels = ['Real'] + model_names
    
    # Plot
    plt.figure(figsize=(10, 6))
    boxplot_colors = [colors.get(label, 'gray') for label in labels]  # Use your defined color mapping
    box = plt.boxplot(data, patch_artist=True, labels=labels)
    
    # Apply colors to the box plots
    for patch, color in zip(box['boxes'], boxplot_colors):
        patch.set_facecolor(color)
    
    plt.title('Distribution of Primary-to-Secondary Peak Intensity Ratios')
    plt.ylabel('Ratio of secondary to primary peak')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    
def old_plot_peak_histograms(peak_intensity_stats_real, peak_intensity_stats_produced, model_names, prodtype, save_path):
    # Ensure real primary and secondary peaks are 1D arrays
    real_primary_peaks = peak_intensity_stats_real[:, 0].flatten().numpy()
    real_secondary_peaks = peak_intensity_stats_real[:, 1].flatten().numpy()

    # Primary Peaks Histogram
    plt.figure(figsize=(10, 6))

    # Plot real primary peaks
    plt.hist(real_primary_peaks, bins=30, alpha=0.5, label='Real primary peak', color=colors['Real'], density=True)

    # Plot each model's produced primary peaks
    for i in range(len(peak_intensity_stats_produced)):
        prod_primary_peaks = peak_intensity_stats_produced[i][:, 0].flatten().numpy()  # Ensure this is a 1D array
        plt.hist(prod_primary_peaks, bins=30, alpha=0.5, label=f'{prodtype} {model_names[i]} primary peak', color=colors.get(model_names[i], 'gray'), density=True)

    plt.title('Histogram of Primary Peak Intensities')
    plt.xlabel('Peak Intensity')
    plt.yscale('log')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}_primary.png')
    plt.close()

    # Secondary Peaks Histogram
    plt.figure(figsize=(10, 6))

    # Plot real secondary peaks
    plt.hist(real_secondary_peaks, bins=30, alpha=0.5, label='Real secondary peak', color=colors['Real'], density=True)

    # Plot each model's produced secondary peaks
    for i in range(len(peak_intensity_stats_produced)):
        prod_secondary_peaks = peak_intensity_stats_produced[i][:, 1].flatten().numpy()  # Ensure this is a 1D array
        plt.hist(prod_secondary_peaks, bins=30, alpha=0.5, label=f'{prodtype} {model_names[i]} secondary peak', color=colors.get(model_names[i], 'gray'), density=True)

    plt.title('Histogram of Secondary Peak Intensities')
    plt.xlabel('Peak Intensity')
    plt.yscale('log')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}_secondary.png')
    plt.close()

def plot_peak_histograms_combined(peak_intensity_stats_real, peak_intensity_stats_produced, model_names, prodtype, save_path):
    # Extract real primary and secondary peak intensities
    real_primary_peaks = np.array(peak_intensity_stats_real['main_peak_value']).flatten()
    real_secondary_peaks = np.array(peak_intensity_stats_real['second_peak_value']).flatten()  


    # Store produced primary and secondary peak intensities per model
    produced_primary_peaks = {}
    produced_secondary_peaks = {}
    
    for i, model_name in enumerate(model_names):
        prod_primary_peaks = peak_intensity_stats_produced[i]['main_peak_value'].flatten().numpy()
        prod_secondary_peaks = peak_intensity_stats_produced[i]['second_peak_value'].flatten().numpy()

        produced_primary_peaks[model_name] = prod_primary_peaks
        produced_secondary_peaks[model_name] = prod_secondary_peaks

    # Combined intensities for bin calculation
    combined_primary_intensities = np.concatenate([real_primary_peaks] + list(produced_primary_peaks.values()))
    combined_secondary_intensities = np.concatenate([real_secondary_peaks] + list(produced_secondary_peaks.values()))

    # Call the provided plotting function for both primary and secondary peaks
    plot_intensity_histograms_individually(real_primary_peaks, produced_primary_peaks, combined_primary_intensities, model_names, 
                                           title='Histogram of Primary Peak Intensities with RMAE', 
                                           xlabel='Primary Peak Intensity', 
                                           save_path=f'{save_path}_primary.png')

    plot_intensity_histograms_individually(real_secondary_peaks, produced_secondary_peaks, combined_secondary_intensities, model_names, 
                                           title='Histogram of Secondary Peak Intensities with RMAE', 
                                           xlabel='Secondary Peak Intensity', 
                                           save_path=f'{save_path}_secondary.png')


def plot_peak_ratio_heatmap(peak_intensity_stats_real, peak_intensity_stats_produced, model_names, prodtype, save_path):
    # Calculate the ratio of primary to secondary peaks for real data
    real_ratios = np.array(peak_intensity_stats_real['main_peak_value']) / np.clip(peak_intensity_stats_real['second_peak_value'], 1e-8, None)
    all_ratios = [real_ratios]

    # Calculate the ratios for produced
    prod_ratios = []
    for prod_stats in peak_intensity_stats_produced:
        prod_ratio = prod_stats['main_peak_value'] / np.clip(prod_stats['second_peak_value'], 1e-8, None)
        prod_ratios.append(prod_ratio)
        all_ratios.append(prod_ratio)

    min_len = min(len(ratio) for ratio in all_ratios)     # Find the minimum length among all arrays
    all_ratios = [ratio[:min_len] for ratio in all_ratios]
    data = np.column_stack(all_ratios)
    log_data = np.log10(data + 1)
    labels = ['Real'] + [f'{name}_{prodtype}' for name in model_names]  # Adjust labels

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(log_data, annot=False, cmap='coolwarm', cbar=True, xticklabels=labels, yticklabels=False)
    plt.title('Secondary-to-Primary Peak Intensity Ratios')
    plt.xlabel('Model')
    plt.ylabel('Image Index')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



    
def plot_radial_intensity(original_radial_intensity, rec_radial_intensity_list, models_radial_intensity, model_names, title="Radial Intensity", save=True, save_path='./generator/eval/radial.png'):
    print("Plotting radial intensity")

    # Initialize storage for radial intensities by encoder
    all_radial_intensities = {}
    all_reconstructed_intensities = {}

    # Collect and group radial intensities by encoder for generated images
    for model_name, radial_intensity in zip(model_names, models_radial_intensity):
        encoder = model_name.split('_')[0]  # Use the first part of the model name as the encoder identifier
        if encoder not in all_radial_intensities:
            all_radial_intensities[encoder] = []
        all_radial_intensities[encoder].append(radial_intensity)
    
    # Collect and group radial intensities by encoder for reconstructed images
    for model_name, rec_radial_intensity in zip(model_names, rec_radial_intensity_list):
        encoder = model_name.split('_')[0]  # Use the first part of the model name as the encoder identifier
        if encoder not in all_reconstructed_intensities:
            all_reconstructed_intensities[encoder] = []
        all_reconstructed_intensities[encoder].append(rec_radial_intensity)

    plt.figure(figsize=(10, 6))

    # Plot original (real) radial intensity as a solid black line
    plt.plot(original_radial_intensity, label='Real (solid black)', color='black', linestyle='-', linewidth=2)

    # Plot generated radial intensities with a dotted black line
    for encoder, radial_intensities in all_radial_intensities.items():
        mean_radial_intensity = np.mean(radial_intensities, axis=0)
        std_radial_intensity = np.std(radial_intensities, axis=0)  # Calculate standard deviation
        rmae = calculate_rmae(original_radial_intensity, mean_radial_intensity)
        color = colors.get(encoder, 'gray')  # Use a default color if encoder is not in colors dict

        # Plot the mean generated radial intensity with a dotted black line
        plt.plot(mean_radial_intensity, label=f'{encoder} Generated (dotted black, RMAE={rmae:.3f})', color='black', linestyle=':', alpha=0.5)

        # Add error bars for the standard deviation
        plt.errorbar(range(len(mean_radial_intensity)), mean_radial_intensity, yerr=std_radial_intensity, fmt='none', ecolor=color, capsize=2)

    # Plot reconstructed radial intensities with dashed black lines
    for encoder, rec_radial_intensities in all_reconstructed_intensities.items():
        mean_rec_radial_intensity = np.mean(rec_radial_intensities, axis=0)
        std_rec_radial_intensity = np.std(rec_radial_intensities, axis=0)  # Calculate standard deviation
        rmae = calculate_rmae(original_radial_intensity, mean_rec_radial_intensity)
        color = colors.get(encoder, 'gray')  # Use a default color if encoder is not in colors dict

        # Plot the mean reconstructed radial intensity with a dashed black line
        plt.plot(mean_rec_radial_intensity, label=f'{encoder} Reconstructed (dashed black, RMAE={rmae:.3f})', color='black', linestyle='--', alpha=0.5)

        # Add error bars for the standard deviation
        plt.errorbar(range(len(mean_rec_radial_intensity)), mean_rec_radial_intensity, yerr=std_rec_radial_intensity, fmt='none', ecolor=color, capsize=2)

    # Set axis labels, title, and grid
    plt.xlabel('Distance from Center')
    plt.ylabel('Average Pixel Intensity')
    plt.title(title)
    plt.grid(True)

    # Create a custom legend in two columns
    handles, labels = plt.gca().get_legend_handles_labels()

    # First column: Solid black (real), dashed black (reconstructed), dotted black (generated)
    real_handle = plt.Line2D([], [], color='black', linestyle='-', label='Real')
    rec_handle = plt.Line2D([], [], color='black', linestyle='--', label='Reconstructed')
    gen_handle = plt.Line2D([], [], color='black', linestyle=':', label='Generated')

    # Second column: Encoders and their associated colors
    encoder_handles = [plt.Line2D([], [], color=colors.get(encoder, 'gray'), label=f'{encoder}') for encoder in all_radial_intensities.keys()]

    # Combine the two columns
    plt.legend(handles=[real_handle, rec_handle, gen_handle] + encoder_handles, loc='upper right', ncol=2)

    # Save or show the plot
    plt.savefig(save_path) if save else plt.show()
    plt.close()
    
    
def plot_stacked_images(real_images, reconstructed_images_list, generated_images_list, model_names, save=True, save_path='./generator/eval/stacked_images.png'):
    print("Plotting stacked images")
    
    # Only keep the first occurrence of each model
    seen_models = set()
    unique_indices = []
    for i, name in enumerate(model_names):
        if name not in seen_models:
            seen_models.add(name)
            unique_indices.append(i)
    
    # Update the model names, generated, and reconstructed images list
    model_names = [model_names[i] for i in unique_indices]
    generated_images_list = [generated_images_list[i] for i in unique_indices]
    reconstructed_images_list = [reconstructed_images_list[i] for i in unique_indices]
    
    num_models = len(generated_images_list)
    num_images_to_use = min(len(real_images), *[len(gen_images) for gen_images in generated_images_list])

    # Select the appropriate number of real, generated, and reconstructed images
    real_images = real_images[:num_images_to_use]
    generated_images_list = [gen_images[:num_images_to_use] for gen_images in generated_images_list]
    reconstructed_images_list = [rec_images[:num_images_to_use] for rec_images in reconstructed_images_list]

    # Calculate the average pixel intensity for each position
    real_avg_intensity = np.mean([img.cpu().numpy().squeeze() for img in real_images], axis=0)
    generated_avg_intensities = [np.mean([img.cpu().numpy().squeeze() for img in gen_images], axis=0) for gen_images in generated_images_list]
    reconstructed_avg_intensities = [np.mean([img.cpu().numpy().squeeze() for img in rec_images], axis=0) for rec_images in reconstructed_images_list]

    # Determine the vmin and vmax based on the data
    all_intensities = [real_avg_intensity] + generated_avg_intensities + reconstructed_avg_intensities
    vmin = min(np.min(intensity) for intensity in all_intensities)
    vmax = max(np.max(intensity) for intensity in all_intensities)

    # Create a layout (all rows with equal height ratios)
    fig, axes = plt.subplots(3, num_models, figsize=(num_models * 2.3, 6), gridspec_kw={'height_ratios': [2, 2, 2], 'wspace': 0, 'hspace': 0})  # Adjusted height ratios
    norm = plt.Normalize(vmin=vmin, vmax=vmax)


    # Plot the heatmap for the real image in the first column of the top row
    im = axes[0, 0].imshow(real_avg_intensity, cmap='viridis', norm=norm)
    axes[0, 0].axis('off')

    # Remove the remaining empty plots in the first row
    for i in range(1, num_models):
        axes[0, i].remove()  # Remove instead of turning off the axis

    # Plot heatmaps for reconstructed and generated images
    for i, (rec_avg_intensity, gen_avg_intensity, name) in enumerate(zip(reconstructed_avg_intensities, generated_avg_intensities, model_names)):
        axes[1, i].imshow(rec_avg_intensity, cmap='viridis', norm=norm)
        axes[1, i].axis('off')
        axes[2, i].imshow(gen_avg_intensity, cmap='viridis', norm=norm)
        axes[2, i].axis('off')
        axes[2, i].text(0.5, -0.15, name, va='center', ha='center', fontsize=10, transform=axes[2, i].transAxes)

    # Set row titles rotated by 90 degrees to the left of the first column
    axes[0, 0].text(-0.4, 0.5, 'Original', va='center', ha='center', fontsize=12, rotation=90, transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.4, 0.5, 'Reconstructed', va='center', ha='center', fontsize=12, rotation=90, transform=axes[1, 0].transAxes)
    axes[2, 0].text(-0.4, 0.5, 'Generated', va='center', ha='center', fontsize=12, rotation=90, transform=axes[2, 0].transAxes)

    # Adjust the layout to minimize whitespace and fit images uniformly
    plt.subplots_adjust(wspace=0, hspace=0)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Manually add an axis for the colorbar, adjusted for less whitespace
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Average Pixel Intensity')
    
    plt.savefig(save_path) if save else plt.show()
    plt.close()


def old_plot_stacked_images(real_images, reconstructed_images_list, generated_images_list, model_names, save=True, save_path='./generator/eval/stacked_images.png'):
    print("Plotting stacked images")
    
    # Only keep the first occurrence of each model
    seen_models = set()
    unique_indices = []
    for i, name in enumerate(model_names):
        if name not in seen_models:
            seen_models.add(name)
            unique_indices.append(i)
    
    # Update the model names, generated, and reconstructed images list
    model_names = [model_names[i] for i in unique_indices]
    generated_images_list = [generated_images_list[i] for i in unique_indices]
    reconstructed_images_list = [reconstructed_images_list[i] for i in unique_indices]
    
    num_models = len(generated_images_list)
    
    # Determine the number of images to use based on the smallest set
    num_images_to_use = min(len(real_images), *[len(gen_images) for gen_images in generated_images_list])

    # Select the appropriate number of real, generated, and reconstructed images
    real_images = real_images[:num_images_to_use]
    generated_images_list = [gen_images[:num_images_to_use] for gen_images in generated_images_list]
    reconstructed_images_list = [rec_images[:num_images_to_use] for rec_images in reconstructed_images_list]

    # Calculate the average pixel intensity for each position
    real_avg_intensity = np.mean([img.cpu().numpy().squeeze() for img in real_images], axis=0)
    generated_avg_intensities = [np.mean([img.cpu().numpy().squeeze() for img in gen_images], axis=0) for gen_images in generated_images_list]
    reconstructed_avg_intensities = [np.mean([img.cpu().numpy().squeeze() for img in rec_images], axis=0) for rec_images in reconstructed_images_list]

    # Determine the vmin and vmax based on the data
    all_intensities = [real_avg_intensity] + generated_avg_intensities + reconstructed_avg_intensities
    vmin = min(np.min(intensity) for intensity in all_intensities)
    vmax = max(np.max(intensity) for intensity in all_intensities)

    # Create a 3-row layout (top: original, middle: reconstructed, bottom: generated)
    fig, axes = plt.subplots(3, num_models, figsize=(20, 10))  # 3 rows: original, reconstructed, generated

    # Normalize the color based on intensity values
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot heatmap for real images in the first column of the top row
    im = axes[0, 0].imshow(real_avg_intensity, cmap='viridis', norm=norm)
    axes[0, 0].axis('off')

    # Plot heatmaps for reconstructed and generated images
    for i, (rec_avg_intensity, gen_avg_intensity, name) in enumerate(zip(reconstructed_avg_intensities, generated_avg_intensities, model_names)):
        axes[1, i].imshow(rec_avg_intensity, cmap='viridis', norm=norm)
        axes[1, i].axis('off')
        axes[2, i].imshow(gen_avg_intensity, cmap='viridis', norm=norm)
        axes[2, i].axis('off')
        axes[2, i].text(0.5, -0.15, name, va='center', ha='center', fontsize=10, transform=axes[2, i].transAxes)

    # Set row titles rotated by 90 degrees to the left of the first column
    axes[0, 0].text(-0.4, 0.5, 'Original', va='center', ha='center', fontsize=12, rotation=90, transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.4, 0.5, 'Reconstructed', va='center', ha='center', fontsize=12, rotation=90, transform=axes[1, 0].transAxes)
    axes[2, 0].text(-0.4, 0.5, 'Generated', va='center', ha='center', fontsize=12, rotation=90, transform=axes[2, 0].transAxes)

    plt.subplots_adjust(left=0.05, right=0.88, top=0.85, bottom=0.05)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Average Pixel Intensity')

    plt.savefig(save_path) if save else plt.show()
    plt.close()


def plot_intensity_histograms_individually(real_intensity_sum, all_produced_sums, combined_intensities, model_names, title, bottomx=9e-4, xlabel='Intensity sum', bins=None, save=True, save_path='./generator/eval/summedintensityhist.png'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Define histogram bins based on combined intensities
    if bins is None:
        bins = np.histogram_bin_edges(combined_intensities, bins=30)
        
    real_hist, _ = np.histogram(real_intensity_sum, bins=bins, density=True)
    real_hist /= np.sum(real_hist)

    real_std = np.sqrt(real_hist / len(real_intensity_sum))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    capsize = 4  # Length of the horizontal caps

    ax1.errorbar(bin_centers, real_hist, yerr=real_std, fmt=' ', capsize=capsize, label=f'Real (n={len(real_intensity_sum)})', color='black', alpha=0.7, elinewidth=1)
    ax1.step(bins, np.append(real_hist, real_hist[-1]), where='post', color=colors['Real'])

    # Histogram for generated/reconstructed images
    for encoder in set(model_names):
        concatenated_sums = all_produced_sums[encoder]
        gen_hist, _ = np.histogram(concatenated_sums, bins=bins, density=True)
        
        # Normalize the generated histogram to ensure its sum is 1
        gen_hist /= np.sum(gen_hist)
        
        gen_std = np.sqrt(gen_hist / len(concatenated_sums))
        rmae_gen = calculate_rmae(real_hist, gen_hist)
        step_bin_edges = np.repeat(bins, 2)[1:-1]
        step_hist_values = np.repeat(gen_hist, 2)
        ax1.step(bins, np.append(gen_hist, gen_hist[-1]), where='post', color=colors[encoder], linestyle=':', label=f'{encoder} (n={len(concatenated_sums)}, RMAE={rmae_gen:.3f})', alpha=0.7)
        ax1.fill_between(step_bin_edges, step_hist_values, step='pre', color=colors[encoder], alpha=0.3)
        ax1.errorbar(bin_centers, gen_hist, yerr=gen_std, fmt=' ', capsize=capsize, color=colors[encoder], alpha=0.7, elinewidth=1)

        relative_error = 2 * (real_hist - gen_hist) / (real_hist + gen_hist + 1e-10)
        ax2.bar(bin_centers, relative_error, width=(bins[1] - bins[0]), color=colors[encoder], alpha=0.4, label=f'{encoder} Relative Error')

    # Set axis labels, scale, and legend
    ax1.set_yscale('log')
    ax1.set_ylabel('Frequency')
    ax1.set_ylim(bottom=bottomx)
    if 'peak' in save_path:
        ax1.legend(loc='upper left')
    else:
        ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.set_ylabel(r'$ 2 \times \frac{(\mathrm{Real} - \mathrm{Gen})}{\mathrm{Real} + \mathrm{Gen}}$')
    ax2.set_xlabel(xlabel)
    ax2.grid(True)

    # Set the x-limits based on the full range of the combined intensities
    x_limits = [bin_centers[0] - (bins[1] - bins[0]) / 2, bin_centers[-1] + (bins[1] - bins[0]) / 2]
    ax1.set_xlim(x_limits)
    ax2.set_xlim(x_limits)
    ax2.set_ylim(min(-1, np.min(relative_error)), max(1, np.max(relative_error)))

    fig.suptitle(title)
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
    
    # Filter latent representations to ensure they exist for the unique indices
    latent_representations = [latent_representations[i] for i in unique_indices if i < len(latent_representations)]
    
    num_models = len(latent_representations)
    
    if num_models == 1:
        fig, axes = plt.subplots(1, num_models, figsize=(15, 5))
        axes = [axes]  # Convert single Axes object to a list
    else:
        fig, axes = plt.subplots(1, num_models, figsize=(15, 5))

    sc = None  # Initialize `sc` to None

    for i, (latents, name) in enumerate(zip(latent_representations, model_names)):
        # Check for NaN or Inf values in latents
        if not np.all(np.isfinite(latents.cpu().numpy())):
            print(f"Skipping {name} due to invalid latent representation values (NaN or Inf found).")
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
    
    
def plot_latent_distributions(latent_representations, save=True, save_path='./latent_distributions.png'):
    latent_representations = latent_representations.cpu().numpy()
    latent_dim = latent_representations.shape[1]
    
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < latent_dim:
            latent_values = latent_representations[:, i]
            latent_values = latent_values[np.isfinite(latent_values)]
            if len(latent_values) == 0:
                ax.axis('off')
                continue
            ax.hist(latent_values, bins=30, color='blue', alpha=0.7)
            ax.set_title(f'Latent Variable {i+1}')
            if np.isfinite(latent_representations).all():
                ax.set_xlim(min(latent_representations.min(), -3), max(latent_representations.max(), 3))
            else:
                ax.set_xlim(-3, 3)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()



def plot_interpolated_images(interpolated_images, save=True, save_path='./interpolation_graph.png'):
    num_steps = len(interpolated_images)
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
    for i, img in enumerate(interpolated_images):
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path) if save else plt.show()
    plt.close()
    
def calculate_statistics(speed_data):
    means = speed_data.mean(axis=0)
    stds = speed_data.std(axis=0)
    return means, stds


def plot_generation_speed(models, save=True, save_path='./generator/eval/speedhist.png'):
    model_dict = {}

    # Parse the log files and calculate training speeds
    for model_info in models:
        if os.path.exists(model_info["path"]):
            scatter_time_1, scatter_time_2, train_time, epochs = parse_log_file(model_info["path"])
            model_name = model_info["name"]
            num_galaxies = model_info.get("num_galaxies", 1)  # Default to 1 if not provided
            
            if epochs > 0 and train_time > 0:  # Prevent division by zero
                train_speed = num_galaxies / (train_time * epochs)
                
                # Add the train speed to the respective model name
                if model_name not in model_dict:
                    model_dict[model_name] = []
                model_dict[model_name].append(train_speed)

    # Prepare data for plotting
    model_names = list(model_dict.keys())
    mean_speeds = [np.mean(model_dict[name]) for name in model_names]
    std_speeds = [np.std(model_dict[name]) for name in model_names]

    # Plotting
    plt.figure(figsize=(10, 6))
    index = np.arange(len(model_names))

    # Plot bars with error bars
    plt.bar(index, mean_speeds, yerr=std_speeds, capsize=5, label='Training Speed')

    plt.xlabel('Model')
    plt.ylabel('Images per Second per Epoch')
    plt.title('Model Training Speeds with Mean and Standard Deviation')
    plt.xticks(index, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
    
    
def plot_model_times_with_error_bars(models, save=True, save_path='./generator/eval/time_with_errors.png'):
    scatter_times_1 = []
    scatter_times_2 = []
    train_times = []
    model_names = []
    
    # Collecting data for each model
    for model_info in models:
        if os.path.exists(model_info["path"]):
            scatter_time_1, scatter_time_2, train_time, epochs = parse_log_file(model_info["path"])            
            model_name = model_info["name"]
            
            scatter_times_1.append(scatter_time_1)
            scatter_times_2.append(scatter_time_2)
            train_times.append(train_time)
            model_names.append(model_name)

    # Now calculate mean and std for the entire data (not in chunks of 5)
    def calculate_mean_std(data):
        means = []
        stds = []
        for i in range(len(data)):
            means.append(np.mean(data[i]))
            stds.append(np.std(data[i]))
        return means, stds

    # Calculate means and stds for each time type
    scatter_times_1_means, scatter_times_1_stds = calculate_mean_std(scatter_times_1)
    scatter_times_2_means, scatter_times_2_stds = calculate_mean_std(scatter_times_2)
    train_times_means, train_times_stds = calculate_mean_std(train_times)

    # Use all model names (no need to reduce them by 5s)
    reduced_model_names = model_names

    # Ensure the number of model names matches the number of calculated means
    if len(reduced_model_names) != len(scatter_times_1_means):
        raise ValueError("Mismatch between the number of model names and the calculated means. Check your data.")

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.2
    index = np.arange(len(reduced_model_names))

    # Plot with error bars
    ax.bar(index - bar_width, scatter_times_1_means, bar_width, yerr=scatter_times_1_stds, label='Train Scattering Time', capsize=5)
    ax.bar(index, scatter_times_2_means, bar_width, yerr=scatter_times_2_stds, label='Test Scattering Time', capsize=5)
    ax.bar(index + bar_width, train_times_means, bar_width, yerr=train_times_stds, label='Training Time', capsize=5)

    ax.set_xlabel('Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Model Times')
    ax.set_xticks(index)
    ax.set_xticklabels(reduced_model_names, rotation=45)
    ax.legend()

    plt.tight_layout()

    if save:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
    

# Plot generated vs original images
vae_multmod(test_images[:num_display], reconstructed_images_list, model_names, 
            save=True, save_path=f"{save_dir}/reconstructed.png", show_title=False, show_originals=True)
# Ensure tensors are moved to CPU and then convert to numpy for shape printing
if generated_images_list:
    print("Generated images exist")
else:
    print("Error: generated_images_list is empty.")
if reconstructed_images_list:
    print("Reconstructed images exist")
else:
    print("Error: reconstructed_images_list is empty.")
vae_multmod(old_images=None, generated_images=generated_images_list, model_names=model_names,
            save=True, save_path=f"{save_dir}/generated.png", show_title=False, show_originals=False)

# SIMILAR GENERATED IMAGES AND ORIGINAL IMAGES
most_similar_images_list = []
for generated_images in generated_images_list:
    most_similar_images = find_most_similar_images(test_images[:num_display], generated_images)
    most_similar_images_list.append(most_similar_images)
vae_multmod(test_images[:num_display], most_similar_images_list, model_names,
            save=True, save_path=f"{save_dir}/generated_similar.png", show_title=False, show_originals=True)

#COMPONENT PEAK STATISTICS
# Initialize empty lists to store peak intensity statistics for all encoders
all_peak_intensity_stats_rec = []
all_peak_intensity_stats_gen = []

#plot_asymmetry_vs_mse(test_images, reconstructed_images, save_path=f"{save_dir}/asymmetry_{encoder_name}.png")

# Loop through each encoder and accumulate the peak statistics
peaks_info_real = get_main_and_secondary_peaks_with_locations(test_images, threshold=0.1)
plot_images_with_peaks(test_images, peaks_info_real, save_path=f'{save_dir}/peaks_real.png')
for i, encoder_name in enumerate(model_names):
    print("Getting peaks info for encoder: ", encoder_name)
    
    # Get the images for the current encoder
    reconstructed_images = reconstructed_images_list[i]
    generated_images = generated_images_list[i]

    # Filter real, reconstructed, and generated images by peak intensity for the current encoder
    peaks_info_rec = get_main_and_secondary_peaks_with_locations(reconstructed_images)
    peaks_info_gen = get_main_and_secondary_peaks_with_locations(generated_images)

    # Accumulate the statistics for the reconstructed and generated images for all encoders
    all_peak_intensity_stats_rec.append(peaks_info_rec)
    all_peak_intensity_stats_gen.append(peaks_info_gen)
    
    plot_asymmetry_vs_mse(test_images, reconstructed_images, encoder_name, save_path=f"{save_dir}/asymmetry_{encoder_name}.png")
    plot_images_by_asymmetry_bins(test_images, reconstructed_images, save_path=f'{save_dir}/asymmetry_bins_{encoder_name}.png')
    plot_asymmetry_score_histogram(test_images, num_bins=21, save_path=f"{save_dir}/asymmetry_histogram_{encoder_name}.png")
    
    plot_images_with_peaks(reconstructed_images, peaks_info_rec, save_path=f'{save_dir}/peaks_rec_{encoder_name}.png')
    plot_images_with_peaks(generated_images, peaks_info_gen, save_path=f'{save_dir}/peaks_gen_{encoder_name}.png')

# Plot primary and secondary peak statistics for all encoders together
plot_peak_statistics(peaks_info_real, all_peak_intensity_stats_rec, model_names, 
                     title='Primary and Secondary Peak Intensities (Reconstructed vs Real)',
                     save_path=f'{save_dir}/peak_stats_rec.png')
plot_peak_statistics(peaks_info_real, all_peak_intensity_stats_gen, model_names, 
                     title='Primary and Secondary Peak Intensities (Generated vs Real)',
                     save_path=f'{save_dir}/peak_stats_gen.png')

# Plot distribution of primary-to-secondary peak intensity ratios for all encoders
plot_peak_ratio_distribution(peaks_info_real, all_peak_intensity_stats_rec, model_names, 
                             save_path=f'{save_dir}/peak_ratio_dist_rec.png')
plot_peak_ratio_distribution(peaks_info_real, all_peak_intensity_stats_gen, model_names, 
                             save_path=f'{save_dir}/peak_ratio_dist_gen.png')

# Plot histograms for primary and secondary peaks for all encoders together
plot_peak_histograms_combined(peaks_info_real, all_peak_intensity_stats_rec, model_names, prodtype='Reconstructed',
                     save_path=f'{save_dir}/peak_histograms_rec.png')
plot_peak_histograms_combined(peaks_info_real, all_peak_intensity_stats_gen, model_names, prodtype='Generated',
                     save_path=f'{save_dir}/peak_histograms_gen.png')

# Plot heatmap for primary-to-secondary peak intensity ratios for all encoders together
plot_peak_ratio_heatmap(peaks_info_real, all_peak_intensity_stats_rec, model_names, prodtype='Reconstructed',
                        save_path=f'{save_dir}/peak_ratio_rec_heatmap')
plot_peak_ratio_heatmap(peaks_info_real, all_peak_intensity_stats_gen, model_names, prodtype='Generated',
                        save_path=f'{save_dir}/peak_ratio_gen_heatmap')


# PIXEL DISTRIBUTIONS
# Calculate summed intensities and plot
real_intensity_sum, all_reconstructed_sums, real_rec_combinations = calculate_summed_intensities(test_images, reconstructed_images_list, model_names)
plot_intensity_histograms_individually(real_intensity_sum, all_reconstructed_sums, real_rec_combinations, model_names, 
                                       title=f"Reconstructions of {galaxy_classes} coded {num_galaxies}", 
                                       xlabel='Intensity sum', save_path=f"{save_dir}/summedintensitydist_recon.png")
real_intensity_sum, all_generated_sums, real_gen_combinations = calculate_summed_intensities(test_images, generated_images_list, model_names)
plot_intensity_histograms_individually(real_intensity_sum, all_generated_sums, real_gen_combinations, model_names, 
                                       title=f"Generations of {galaxy_classes} coded {num_galaxies}", 
                                       xlabel='Intensity sum', save_path=f"{save_dir}/summedintensitydist_gen.png")

# Calculate peak intensities and plot
real_peak_intensity, all_reconstructed_peaks, real_rec_combinations = calculate_peak_intensities(test_images, reconstructed_images_list, model_names)
plot_intensity_histograms_individually(real_peak_intensity, all_reconstructed_peaks, real_rec_combinations, model_names,
                                       #bins=np.linspace(0.8, 1.0, 30), 
                                       title=f"Reconstructions of {galaxy_classes} coded {num_galaxies}",
                                       xlabel='Peak Intensity', save_path=f"{save_dir}/peakintensitydist_recon.png")
real_peak_intensity, all_generated_peaks, real_gen_combinations = calculate_peak_intensities(test_images, generated_images_list, model_names)
plot_intensity_histograms_individually(real_peak_intensity, all_generated_peaks, real_gen_combinations, model_names, 
                                       #bins=np.linspace(0.7, 1.0, 30),
                                       title=f"Generations of {galaxy_classes} coded {num_galaxies}",
                                       xlabel='Peak Intensity', save_path=f"{save_dir}/peakintensitydist_gen.png")

# Calculate total intensities and plot
real_total_intensity, all_reconstructed_total_intensities, real_rec_combinations = calculate_total_intensities(test_images, reconstructed_images_list, model_names)
plot_intensity_histograms_individually(real_total_intensity, all_reconstructed_total_intensities, real_rec_combinations, model_names, 
                                       title=f"Reconstructions of {galaxy_classes} coded {num_galaxies}",
                                       xlabel='Total Intensity', bottomx=1e-6, save_path=f"{save_dir}/totalintensitydist_recon.png")
real_total_intensity, all_generated_total_intensities, real_gen_combinations = calculate_total_intensities(test_images, generated_images_list, model_names)
plot_intensity_histograms_individually(real_total_intensity, all_generated_total_intensities, real_gen_combinations, model_names, 
                                       title=f"Generations of {galaxy_classes} coded {num_galaxies}",
                                       xlabel='Total Intensity', bottomx=1e-6, save_path=f"{save_dir}/totalintensitydist_gen.png")


# RADIAL INTENSITY COMPARISON
original_radial_intensity = compute_radial_intensity(test_images[:num_display], img_shape)
models_gen_radial_intensity, models_rec_radial_intensity = [], []
for i, generated_images in enumerate(generated_images_list):
    radial_intensity = compute_radial_intensity(generated_images, img_shape)
    models_gen_radial_intensity.append(radial_intensity)
for i, reconstructed_images in enumerate(reconstructed_images_list):
    radial_intensity = compute_radial_intensity(reconstructed_images, img_shape)
    models_rec_radial_intensity.append(radial_intensity)    
title = f'Radial Intensity of {galaxy_classes}'
plot_radial_intensity(original_radial_intensity, models_rec_radial_intensity, models_gen_radial_intensity, model_names, title, save_path=f"{save_dir}/radial.png")
plot_stacked_images(test_images, reconstructed_images_list, generated_images_list, model_names, save_path=f"{save_dir}/stacked_images.png")
plot_generation_speed(models, save_path=f'{save_dir}/speedhist.png')
plot_model_times_with_error_bars(models, save_path=f'{save_dir}/timehist.png')


# LATENT SPACE
latent_representations_combined = torch.cat(latent_representations, dim=0)  # Combine all latent representations
plot_latent_spaces(latent_representations, model_names, save_path=f'{save_dir}/latent_spaces.png')
plot_latent_distributions(latent_representations_combined, save_path=f'{save_dir}/latent_distributions.png')

print("Plots generated and saved.")
