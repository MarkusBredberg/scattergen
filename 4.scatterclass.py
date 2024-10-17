import numpy as np
import torch
import time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from utils.data_loader import load_galaxies, get_classes
from utils.classifiers import Scattering_Classifier, CNN_Classifier, Small_Scattering_Classifier
from utils.training_tools import EarlyStopping, reset_weights
from utils.scatter_reduction import lavg, ldiff
from utils.calc_tools import normalize_to_0_1, cluster_metrics
from utils.plotting import plot_loss, plot_images, plot_histograms
from kymatio.torch import Scattering2D
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.optim import AdamW
import itertools
import os
os.makedirs('./classifier/trained_models', exist_ok=True)

###############################################
################ CONFIGURATION ################
###############################################

classes = get_classes()
galaxy_classes = [10, 11]  # Classes to classify
num_galaxies = 200  # Training size for each class
dataset_portions = [0.5, 1]  # Portions of complete dataset for the accuracy vs dataset size
J, L, order = 4, 8, 2  # Scatter transform parameters
classifier = ["scatterMLP", "normalCNN"][0]  # Choose one model
num_epochs_cuda = 5
num_epochs_cpu = 10
batch_size = 64
lr = 3e-5  # Learning rate
reg = 1e-4 # Regularization parameter
num_experiments = 1
num_folds = 5
img_shape = (1, 128, 128)

FFCV = True # Use five-fold cross-validation
ES = True # Use early stopping
IMGCHECK = False # Check the input images (Tool for control)
SAVEIMGS = False # Save the reconstructed images in tensor format
NORMALISETOPM = False # Normalise to [-1, 1]

#########################################################################################################################

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
    num_epochs = num_epochs_cuda
    print(f"CUDA is available. Setting epochs to {num_epochs}.")
else:
    DEVICE = "cpu"
    num_epochs = num_epochs_cpu
    print(f"CUDA is not available. Setting epochs to {num_epochs}.")

classes = get_classes()


###############################################
########### DATA STORING FUNCTIONS ###############
###############################################


# Initialize these dictionaries with empty lists for each unique combination of subset_size, galaxy_classes, and model_name
def initialize_metrics(metrics, model_name, subset_size, fold, experiment):
    subset_size_str = str(subset_size[0]) if isinstance(subset_size, list) else str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}"] = []
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}"] = []
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}"] = []
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}"] = []
    
# Function to update metrics with the new values  
def update_metrics(metrics, model_name, subset_size, fold, experiment, accuracy, precision, recall, f1):
    subset_size_str = str(subset_size)
    metrics[f"{model_name}_accuracy_{subset_size_str}_{fold}_{experiment}"].append(accuracy)
    metrics[f"{model_name}_precision_{subset_size_str}_{fold}_{experiment}"].append(precision)
    metrics[f"{model_name}_recall_{subset_size_str}_{fold}_{experiment}"].append(recall)
    metrics[f"{model_name}_f1_score_{subset_size_str}_{fold}_{experiment}"].append(f1)
    
def initialize_history(history, model_name, subset_size, fold, experiment):
    if model_name not in history:
        history[model_name] = {}

    loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}"
    val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}"

    # Initialize lists to store losses
    if loss_key not in history[model_name]:
        history[model_name][loss_key] = []
    if val_loss_key not in history[model_name]:
        history[model_name][val_loss_key] = []
        
def initialize_labels(all_true_labels, all_pred_labels, model_name, subset_size, fold, experiment):
    key = f"{model_name}_{subset_size}_{fold}_{experiment}"
    all_true_labels[key] = []
    all_pred_labels[key] = []

            

###############################################
########### PLOTTING FUNCTIONS ################
###############################################
    
def plot_metrics(metrics, dataset_sizes, model_names, num_folds, num_experiments):
    """
    Plots the metrics for each model across different dataset sizes with lines connecting the dots.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_name in model_names:
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            metric_values = {subset_size: [] for subset_size in dataset_sizes}  # Collect metric values for each size
            
            # Collect metric values across folds and experiments
            for subset_size in dataset_sizes:
                for fold in range(num_folds):
                    for experiment in range(num_experiments):
                        metric_key = f"{model_name}_{metric}_{subset_size}_{fold}_{experiment}"
                        if metric_key in metrics and len(metrics[metric_key]) > 0:
                            metric_values[subset_size].extend(metrics[metric_key])

            # Compute mean and std for each subset size
            mean_values = [np.mean(metric_values[size]) for size in dataset_sizes]
            std_values = [np.std(metric_values[size]) for size in dataset_sizes]
            
            # Plot the metric evolution with lines connecting the points
            ax.errorbar(dataset_sizes, mean_values, yerr=std_values, marker='o', linestyle='-', 
                        label=f"{metric.capitalize()}", color=metric_colors[metric])

        # Set up the plot
        ax.set_title(f"Model Performance Across Dataset Sizes for {model_name}", fontsize=16)
        ax.set_xlabel('Dataset Size', fontsize=14)
        ax.set_ylabel('Metric Value', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        ax.set_xscale('log')  # Set log scale for dataset size

    # Ensure the ticks are correctly set with the right fontsize
    plt.xticks(fontsize=16)  # Properly apply fontsize here
    plt.yticks(fontsize=16)  # And here
    plt.tight_layout()

    # Save the metrics plot
    plt.savefig(f"./classifier/{model_name}_all_metrics.png")
    plt.show()  # Optionally show the plot

    
def plot_all_metrics_vs_dataset_size(metrics, dataset_sizes, model_name, num_folds, num_experiments, save_dir='./classifier'):
    """
    Plots accuracy, precision, recall, and F1 score as a function of dataset size.
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots

    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    metric_titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    colors = ['blue', 'green', 'red', 'orange']  # Assign colors for each metric

    for i, metric in enumerate(metric_names):
        row, col = i // 2, i % 2  # Position in 2x2 grid

        for subset_size in dataset_sizes:
            metric_values = []
            for fold in range(num_folds):
                for experiment in range(num_experiments):
                    key = f"{model_name}_{metric}_{subset_size}_{fold}_{experiment}"
                    if key in metrics and len(metrics[key]) > 0:
                        metric_values.extend(metrics[key])  # Collect all values for that key

            mean_metric = np.mean(metric_values)
            std_metric = np.std(metric_values)
            
            # Plot the mean and error bars for each metric
            ax[row, col].errorbar(subset_size, mean_metric, yerr=std_metric, fmt='o-', color=colors[i], label=f"{metric.capitalize()}")
            ax[row, col].set_title(f"{metric_titles[i]} vs Dataset Size", fontsize=14)
            ax[row, col].set_xlabel('Dataset Size', fontsize=12)
            ax[row, col].set_ylabel(f"{metric_titles[i]}", fontsize=12)
            ax[row, col].grid(True)
            ax[row, col].set_xscale('log')  # Log scale for dataset sizes

        ax[row, col].legend(fontsize=10)

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(f"{save_dir}/{model_name}_metrics_vs_dataset_size.png")
    plt.show()

    
def plot_roc_curves(all_true_labels, all_pred_probs, model_name, dataset_sizes, num_folds, num_experiments, galaxy_classes, save_dir='./classifier'):
    """
    Plots the ROC curve for each run (subset, fold, experiment) and saves it.
    """
    for subset_size in dataset_sizes:
        for fold in range(num_folds):
            for experiment in range(num_experiments):
                key = f"{model_name}_{subset_size}_{fold}_{experiment}"
                true_labels = np.array(all_true_labels[key])
                pred_probs = np.array(all_pred_probs[key])

                # Calculate ROC and AUC for class 1 (positive class in binary classification)
                fpr, tpr, _ = roc_curve(true_labels, pred_probs[:, 1])  # Assuming binary classification
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=16)
                ax.set_ylabel('True Positive Rate', fontsize=16)
                ax.set_title(f'ROC Curve - {model_name} \n {subset_size}, Fold {fold}, Experiment {experiment}', fontsize=14)
                ax.legend(loc="lower right")
                
                # Save the ROC plot
                plt.savefig(f'{save_dir}/{model_name}_{galaxy_classes}_{subset_size}_{fold}_{experiment}_roc_curve.png')
                plt.close()


def plot_confusion_matrix(all_true_labels, all_pred_labels, model_name, dataset_sizes, num_folds, num_experiments, galaxy_classes, class_descriptions, save_dir='./classifier'):
    """
    Plots the confusion matrix and saves it for the combined results across folds and experiments.
    """
    for subset_size in dataset_sizes:
        for fold in range(num_folds):
            for experiment in range(num_experiments):
                key = f"{model_name}_{subset_size}_{fold}_{experiment}"
                true_labels = all_true_labels[key]
                pred_labels = all_pred_labels[key]

                cm = confusion_matrix(true_labels, pred_labels, normalize='true')

                # Plot the confusion matrix
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt=".1%", linewidths=.5, square=True, cmap='Blues', ax=ax,
                            xticklabels=class_descriptions, yticklabels=class_descriptions, annot_kws={"size": 16})
                colorbar = ax.collections[0].colorbar
                colorbar.ax.tick_params(labelsize=16)

                # Set labels and title
                accuracy = accuracy_score(true_labels, pred_labels)
                ax.set_title(f'Model: {model_name} \n Total accuracy: {accuracy*100:.2f}%', fontsize=14)
                ax.set_ylabel('True label', fontsize=16)
                ax.set_xlabel('Predicted label', fontsize=16)

                # Save Confusion Matrix
                plt.savefig(f'{save_dir}/{model_name}_{galaxy_classes}_{subset_size}_{fold}_{experiment}_confusion_matrix.png')
                plt.close()
    
def plot_loss(models, dataset_sizes, num_folds, num_experiments, galaxy_classes, num_galaxies, save_dir='./classifier'):
    fig, ax = plt.subplots(figsize=(8, 6))  # Slightly larger figure for clarity

    # Define a cycle of line styles
    line_styles = itertools.cycle(['-', '--', '-.', ':'])  # Cycles through these styles

    for model_name, model_details in models.items():
        for subset_size in dataset_sizes:
            # Get the next line style for each subset size
            linestyle = next(line_styles)
            
            # Plot training loss in blue
            loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}"
            ax.plot(history[model_name][loss_key], color='blue', linestyle=linestyle, label=f"{model_name} training loss (size {subset_size})")
            
            # Plot validation loss in orange
            val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}"
            ax.plot(history[model_name][val_loss_key], color='orange', linestyle=linestyle, label=f"{model_name} validation loss (size {subset_size})")

    # Add title and labels
    ax.set_title('Training and Validation Loss', fontsize=16)
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)

    # Show legend and grid
    ax.legend(fontsize=12, loc='best')  # Adjust the fontsize of the legend and place it at the best location
    ax.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"./classifier/{classifier}_{galaxy_classes}_{num_galaxies}_loss.png")
    plt.close()


###############################################
########### LOOP OVER DATA FOLD ###############
###############################################

# Initialize metric vectors dictionary
metrics = {
    "accuracy": {},
    "precision": {},
    "recall": {},
    "f1_score": {}
}

metric_colors = {
    "accuracy": 'blue',
    "precision": 'green',
    "recall": 'red',
    "f1_score": 'orange'
}

all_true_labels = {}
all_pred_labels = {}
training_times = {}
all_pred_probs = {}

dataset_sizes = [int(num_galaxies * 0.8 * perc) for perc in dataset_portions]
"""for subset_size in dataset_sizes:
    for fold in range(num_folds):
        for experiment in range(num_experiments):
            initialize_metrics(metrics, classifier, subset_size, fold, experiment)"""

for fold in range(5) if FFCV else [6]:
    torch.cuda.empty_cache()
    
    # Define the run name
    desc = ', '.join([classes[c]["description"] for c in galaxy_classes])
    runname = f'{galaxy_classes}_{classifier}_{lr}'
    log_path = f"./classifier/log_{runname}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file = open(log_path, 'w')

    # Load the data
    data = load_galaxies(galaxy_class=galaxy_classes, 
                        fold=fold,
                        img_shape=img_shape, 
                        sample_size=num_galaxies, 
                        process=True, 
                        train=True, 
                        runname=None, 
                        generated=False, 
                        reconstructed=False)
    train_images, train_labels, test_images, test_labels = data
    
    print("Train images shape before filtering:", np.shape(train_images))
    print("Test images shape before filtering:", np.shape(test_images))

    # Check the input data
    print("Train images shape before filtering:", np.shape(train_images))
    if IMGCHECK:
        plot_images(train_images, train_labels)
        plot_histograms(train_images, test_images)


    # Prepare input data
    if 'MLP' in classifier:
        scattering = Scattering2D(J=J, L=L, shape=img_shape[1:], max_order=order).to(DEVICE)    
        def compute_scattering_coeffs(images):
            print("Computing scattering coefficients...")
            scat_batch_size = 16
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
        
        if 'lavg' in classifier:
            train_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([lavg(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])
        elif 'ldiff' in classifier:
            train_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in train_scat_coeffs])
            test_scat_coeffs = torch.stack([ldiff(coeff, J=J, L=L, m=order)[0] for coeff in test_scat_coeffs])

    # Normalize train and test images to [0, 1]
    train_images = normalize_to_0_1(train_images)
    test_images = normalize_to_0_1(test_images)

    if NORMALISETOPM:
        # If NORMALISETOPM is True, normalize to [-1, 1]
        train_images = train_images * 2 - 1
        test_images = test_images * 2 - 1

    # Handle scattering coefficients normalization in a similar way
    if 'MLP' in classifier:
        train_scat_coeffs = normalize_to_0_1(train_scat_coeffs)
        test_scat_coeffs = normalize_to_0_1(test_scat_coeffs)

        if NORMALISETOPM:
            train_scat_coeffs = train_scat_coeffs * 2 - 1
            test_scat_coeffs = test_scat_coeffs * 2 - 1


    #Check input after renormalisation and filtering  
    if IMGCHECK: 
        plot_images(train_images, train_labels)
        plot_histograms(train_images, test_images)
        
    # Flatten the scattering coefficients
    scatdim = int(np.prod(train_scat_coeffs.shape[1:]))
    train_scat_coeffs = train_scat_coeffs.view(-1, scatdim)
    test_scat_coeffs = test_scat_coeffs.view(-1, scatdim)


    # Reshape labels to one-hot encoding
    train_labels = torch.nn.functional.one_hot(train_labels, num_classes=len(galaxy_classes)).float()
    test_labels = torch.nn.functional.one_hot(test_labels, num_classes=len(galaxy_classes)).float()


    if 'MLP' in classifier: #Double dataset for convenience for dual model in training loop
        train_dataset = TensorDataset(train_scat_coeffs, train_labels)
        test_dataset = TensorDataset(test_scat_coeffs, test_labels)
    else: 
        train_dataset = TensorDataset(train_images, train_labels) 
        test_dataset = TensorDataset(test_images, test_labels) 
        

    # Create the data loaders
    def custom_collate(batch):
        if len(batch) < 3: # Drop the batch if it contains fewer than 3 images
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate, drop_last=True)

            
    ###############################################
    ############# DEFINE MODEL ####################
    ###############################################
    
    scatshape = int(np.prod(train_scat_coeffs.shape[1:]))
    print("Shape of scattering coefficients:", train_scat_coeffs.shape)
    print("Number of classes:", len(galaxy_classes))

    # Selection of model
    if classifier == "scatterMLP":
        models = {"scatterMLP": {"model": Scattering_Classifier(input_size=scatshape, l1=64, l2=128, num_classes=len(galaxy_classes)).to(DEVICE)}}
    elif classifier == "smallSTMLP":
        models = {"smallSTMLP": {"model": Small_Scattering_Classifier(input_size=scatshape, l1=128, num_classes=len(galaxy_classes)).to(DEVICE)}}
    elif classifier == "normalCNN":
        models = {"normalCNN": {"model": CNN_Classifier(num_classes=len(galaxy_classes)).to(DEVICE)}}
    else:
        raise ValueError("Model not found. Please select one of 'scatterMLP', 'smallSTMLP', or 'normalCNN'.")

    # Apply summary to each model individually
    for model_name, model_details in models.items():
        print(f"Summary for {model_name}:")
        if model_name in ["scatterMLP", "smallSTMLP"]:
            summary(model_details["model"], input_size=(scatdim,), device=DEVICE)
        else:
            summary(model_details["model"], input_size=img_shape, device=DEVICE)

    ###############################################
    ############### TRAINING LOOP #################
    ###############################################

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(models[model_name]["model"].parameters(), lr=lr, weight_decay=reg)
    history = {}
    # Initialize an empty dictionary for training times
    training_times = {}

    # Loop over models and dataset sizes
    for model_name, model_details in models.items():
        print(f"Training {model_name} model...")
        model = model_details["model"].to(DEVICE)

        for subset_size in dataset_sizes:
            if subset_size not in training_times:
                training_times[subset_size] = {}  # Initialize the times for this subset size

            for experiment in range(num_experiments):
                for fold in range(num_folds):
                    initialize_history(history, model_name, subset_size, fold, experiment)
                    initialize_metrics(metrics, model_name, subset_size, fold, experiment)
                    initialize_labels(all_true_labels, all_pred_labels, model_name, subset_size, fold, experiment)

                    start_time = time.time()
                    model.apply(reset_weights)

                    # Create subset data loader
                    subset_indices = list(range(subset_size))
                    subset_train_dataset = Subset(train_dataset, subset_indices)
                    subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

                    early_stopping = EarlyStopping(patience=10, verbose=False) if ES else None

                    for epoch in tqdm(range(num_epochs), desc=f'Training with dataset size {subset_size}'):
                        model.train()
                        total_loss = 0
                        total_images = 0

                        for images, labels in subset_train_loader:
                            images, labels = images.to(DEVICE), labels.to(DEVICE)
                            optimizer.zero_grad()
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item() * images.size(0)
                            total_images += images.size(0)

                        average_loss = total_loss / total_images
                        loss_key = f"{model_name}_loss_{subset_size}_{fold}_{experiment}"
                        history[model_name][loss_key].append(average_loss)

                        # Validation loop
                        model.eval()
                        val_total_loss = 0
                        val_total_images = 0
                        correct = 0

                        with torch.no_grad():
                            for images, labels in test_loader:
                                images, labels = images.to(DEVICE), labels.to(DEVICE)
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                                val_total_loss += loss.item() * images.size(0)
                                val_total_images += images.size(0)
                                _, predicted = torch.max(outputs, 1)
                                true_labels = torch.argmax(labels, dim=1)
                                correct += (predicted == true_labels).sum().item()

                        val_average_loss = val_total_loss / val_total_images
                        val_loss_key = f"{model_name}_val_loss_{subset_size}_{fold}_{experiment}"
                        history[model_name][val_loss_key].append(val_average_loss)

                        if ES: # Early stopping
                            early_stopping(val_average_loss, model, f'./classifier/trained_models/{model_name}_best_model.pth')
                            if early_stopping.early_stop:
                                print("Early stopping")
                                break

                            
                    # Final evaluation after training
                    with torch.no_grad():
                        # Get raw outputs (logits) and apply sigmoid for binary classification
                        if 'MLP' in classifier:
                            outputs = model(test_scat_coeffs.to(DEVICE))
                        else:
                            outputs = model(test_images.to(DEVICE))

                        pred_probs = torch.sigmoid(outputs).cpu().numpy()  # Use sigmoid for binary classification

                        true_labels = torch.argmax(test_labels, dim=1).cpu().numpy()
                        pred_labels = np.argmax(pred_probs, axis=1)

                        # Store the labels and probabilities for this particular fold, experiment, and subset
                        key = f"{model_name}_{subset_size}_{fold}_{experiment}"
                        all_true_labels[key].extend(true_labels)
                        all_pred_labels[key].extend(pred_labels)  # For label predictions
                        all_pred_probs[key] = pred_probs  # Store the predicted probabilities

                        # Now you can safely calculate the metrics
                        accuracy = accuracy_score(true_labels, pred_labels)
                        precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
                        recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
                        f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

                        update_metrics(metrics, model_name, subset_size, fold, experiment, accuracy, precision, recall, f1)


                    # Track training time
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken to train the model for fold {fold}: {elapsed_time:.2f} seconds")

                    # Save the training time for the current fold
                    if fold not in training_times[subset_size]:
                        training_times[subset_size][fold] = []
                    training_times[subset_size][fold].append(elapsed_time)

                    file.write(f"Time taken to train the model on fold {fold}: {elapsed_time:.2f} seconds \n")

                    
            # Calculate cluster metrics after generating images
            if model_name == "scatterMLP" or model_name == "smallSTMLP": 
                generated_features = model(test_scat_coeffs.to(DEVICE)).cpu().detach().numpy()
            else:
                generated_features = model(test_images.to(DEVICE)).cpu().detach().numpy()

            cluster_error, cluster_distance, cluster_std_dev = cluster_metrics(generated_features, n_clusters=13)

            print(f"Cluster Error: {cluster_error}")
            print(f"Cluster Distance: {cluster_distance}")
            print(f"Cluster Standard Deviation: {cluster_std_dev}")

        # Save the trained model
        model_save_path = f'./classifier/trained_models/{model_name}_model.pth'
        torch.save(model.state_dict(), model_save_path)


###############################################
############ PLOTS AFTER ALL FOLDS ############
###############################################

class_descriptions = [cls['description'] for cls in classes if cls['tag'] in galaxy_classes]
plot_confusion_matrix(all_true_labels, all_pred_labels, model_name, dataset_sizes, num_folds, num_experiments, galaxy_classes, class_descriptions)
plot_roc_curves(all_true_labels, all_pred_probs, model_name, dataset_sizes, num_folds, num_experiments, galaxy_classes, save_dir='./classifier')
plot_loss(models, dataset_sizes, num_folds, num_experiments, galaxy_classes, num_galaxies, save_dir='./classifier')
plot_all_metrics_vs_dataset_size(metrics, dataset_sizes, model_name, num_folds, num_experiments, save_dir='./classifier')
plot_metrics(metrics, dataset_sizes, [model_name], num_folds, num_experiments)

# Print metrics summary for each subset size
print("Metrics:")
for metric in ["accuracy", "precision", "recall", "f1_score"]:
    print(f"{metric.capitalize()}:")
    for subset_size in dataset_sizes:
        metric_values = []
        for fold in range(num_folds):
            for experiment in range(num_experiments):
                key = f"{classifier}_{metric}_{subset_size}_{fold}_{experiment}"
                metric_values.append(metrics[key])

        mean = np.mean(metric_values)
        std = np.std(metric_values)

        print(f"Subset size {subset_size}: {mean:.4f} ± {std:.4f}")

# Print the training times for each fold and subset size with mean and std
print("\nTraining Times:")
for subset_size in training_times:
    times = []
    print(f"Subset size {subset_size}:")
    for fold in training_times[subset_size]:
        elapsed_times = training_times[subset_size][fold]  # This is a list of times
        for elapsed_time in elapsed_times:  # Iterate over the list of times
            times.append(elapsed_time)
            print(f"  Fold {fold}: {elapsed_time:.2f} seconds")


    # Calculate mean and standard deviation for each subset size
    mean_time = np.mean(times)
    std_time = np.std(times)

    # Print mean and std
    print(f"  Mean training time: {mean_time:.2f} seconds ± {std_time:.2f} seconds\n")
    

