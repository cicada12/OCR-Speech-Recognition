import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.kde.kde import KDE
from models.gmm.gmm import GMM
import librosa
import seaborn as sns
from hmmlearn import hmm
import warnings
import random



# Set random seed for reproducibility
np.random.seed(42)

# Large circle with fewer points in the center
num_points_large = 3000
radius_large = np.random.uniform(0, 1, num_points_large) ** 0.5 * 2.1  # Adjust to make the center less dense
angles_large = np.random.uniform(0, 2 * np.pi, num_points_large)
x_large = radius_large * np.cos(angles_large)
y_large = radius_large * np.sin(angles_large)

# Small dense circle
num_points_small = 500
radius_small = np.random.normal(0.2, 0.1, num_points_small)  # Small radius with a little noise
angles_small = np.random.uniform(0, 2 * np.pi, num_points_small)
x_small = 1 + radius_small * np.cos(angles_small)  # Centered around (1, 1)
y_small = 1 + radius_small * np.sin(angles_small)

# Combine both circles
x = np.concatenate((x_large, x_small))
y = np.concatenate((y_large, y_small))

# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=1, color='black', alpha=0.5)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("Original Data")
plt.grid(True)
plt.show()


# Create the dataset
data = np.vstack((x, y)).T

# Fit KDE model
kde = KDE(bandwidth=0.3, kernel='gaussian')
kde.fit(data)

# Fit GMM model with 2 components using the custom GMM class
gmm_2 = GMM(n_components=2, random_state=42)
gmm_2.fit(data)

# Generate grid for visualization
x_vals = np.linspace(-4, 4, 100)
y_vals = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_vals, y_vals)
grid = np.array([X.ravel(), Y.ravel()]).T

# Evaluate KDE using score_samples method
log_dens_kde = kde.score_samples(grid)
Z_kde = np.exp(log_dens_kde).reshape(100, 100)

# Evaluate GMM with 2 components using getMembership method
# For density estimation, we take the sum of component probabilities weighted by component weights
membership_probs = gmm_2.getMembership(grid)
log_dens_gmm_2 = np.log(np.sum(membership_probs * gmm_2.weights, axis=1))
Z_gmm_2 = np.exp(log_dens_gmm_2).reshape(100, 100)

# Plot KDE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, Z_kde, levels=30, cmap='viridis')
plt.scatter(x, y, s=1, color='white', alpha=0.5)
plt.title("KDE Density Estimation")
plt.colorbar()

# Plot GMM with 2 components
plt.subplot(1, 2, 2)
plt.contourf(X, Y, Z_gmm_2, levels=30, cmap='viridis')
plt.scatter(x, y, s=1, color='white', alpha=0.5)
plt.title("GMM Density Estimation (2 components)")
plt.colorbar()

plt.show()


# Flatten density values for color-coding each data point
Z_kde_flat = np.exp(kde.score_samples(data))
Z_gmm_2_flat = np.exp(np.log(np.sum(gmm_2.getMembership(data) * gmm_2.weights, axis=1)))

# Plot KDE Density Estimation with Color-Coded Scatter Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=Z_kde_flat, cmap='viridis', s=5, alpha=0.8)
plt.colorbar(label='Density')
plt.title("KDE Density Estimation")

# Plot GMM Density Estimation with Color-Coded Scatter Plot
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=Z_gmm_2_flat, cmap='viridis', s=5, alpha=0.8)
plt.colorbar(label='Density')
plt.title("GMM Density Estimation (2 components)")

plt.show()


# Fit GMM model with 3 components using the custom GMM class
gmm_3 = GMM(n_components=3, random_state=42)
gmm_3.fit(data)

# Fit GMM model with 4 components using the custom GMM class
gmm_4 = GMM(n_components=4, random_state=42)
gmm_4.fit(data)

# Evaluate GMM with 2 components using getMembership method
# For density estimation, we take the sum of component probabilities weighted by component weights
membership_probs = gmm_3.getMembership(grid)
log_dens_gmm_3 = np.log(np.sum(membership_probs * gmm_3.weights, axis=1))
Z_gmm_3 = np.exp(log_dens_gmm_3).reshape(100, 100)

membership_probs = gmm_4.getMembership(grid)
log_dens_gmm_4 = np.log(np.sum(membership_probs * gmm_4.weights, axis=1))
Z_gmm_4 = np.exp(log_dens_gmm_4).reshape(100, 100)

plt.figure(figsize=(12, 5))

# Plot GMM with 3 components
plt.subplot(1, 2, 1)
plt.contourf(X, Y, Z_gmm_3, levels=30, cmap='viridis')
plt.scatter(x, y, s=1, color='white', alpha=0.5)
plt.title("GMM Density Estimation (3 components)")
plt.colorbar()


# Plot GMM with 4 components
plt.subplot(1, 2, 2)
plt.contourf(X, Y, Z_gmm_4, levels=30, cmap='viridis')
plt.scatter(x, y, s=1, color='white', alpha=0.5)
plt.title("GMM Density Estimation (4 components)")
plt.colorbar()

plt.show()


# Evaluate GMM with 3 components for density estimation
membership_probs = gmm_3.getMembership(data)
Z_gmm_3_flat = np.exp(np.log(np.sum(membership_probs * gmm_3.weights, axis=1)))

# Evaluate GMM with 4 components for density estimation
membership_probs = gmm_4.getMembership(data)
Z_gmm_4_flat = np.exp(np.log(np.sum(membership_probs * gmm_4.weights, axis=1)))

plt.figure(figsize=(12, 5))

# Plot GMM with 3 components (Color-Coded Scatter Plot)
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=Z_gmm_3_flat, cmap='viridis', s=5, alpha=0.8)
plt.colorbar(label='Density')
plt.title("GMM Density Estimation (3 components)")

# Plot GMM with 4 components (Color-Coded Scatter Plot)
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=Z_gmm_4_flat, cmap='viridis', s=5, alpha=0.8)
plt.colorbar(label='Density')
plt.title("GMM Density Estimation (4 components)")

plt.show()


# Suppress warnings for hmmlearn
warnings.filterwarnings("ignore")

# Set path to the dataset
dataset_path = '../../data/external/spoken_digits'

# Extract and visualize MFCCs for a sample file
def extract_mfcc_(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # n_mfcc can be adjusted as needed
    return mfcc

# Example of extracting MFCCs for a single file and visualizing them
file_path = os.path.join(dataset_path, '0_jackson_0.wav')  # Adjust with a sample path
mfcc = extract_mfcc_(file_path)

# Visualize the MFCCs as a heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(mfcc, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.title("MFCC Heatmap for Sample Audio")
plt.xlabel("Time (frames)")
plt.ylabel("MFCC Coefficients")
plt.show()

# Split dataset paths into train, validation, and test
def split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1):
    all_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
    random.shuffle(all_files)
    
    total = len(all_files)
    train_files = all_files[:int(total * train_ratio)]
    val_files = all_files[int(total * train_ratio):int(total * (train_ratio + val_ratio))]
    test_files = all_files[int(total * (train_ratio + val_ratio)):]
    
    return train_files, val_files, test_files

# Extract MFCC features
def extract_mfcc(file_path, n_mfcc=13, n_fft=512):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    return mfcc.T  # Transpose for HMM compatibility

# Prepare data by digit label
def prepare_data(file_list, dataset_path):
    data = {}
    for file_name in file_list:
        digit = int(file_name.split('_')[0])  # Extract digit label from filename
        file_path = os.path.join(dataset_path, file_name)
        mfcc_features = extract_mfcc(file_path)
        if digit not in data:
            data[digit] = []
        data[digit].append(mfcc_features)
    return data


# Initialize HMM models and train them
def train_hmm_models(train_data):
    models = {}
    for digit, features in train_data.items():
        model = hmm.GaussianHMM(n_components=4, covariance_type='diag', n_iter=100)  # Adjust n_components as needed
        X = np.vstack(features)  # Stack all MFCC features for this digit
        lengths = [len(f) for f in features]  # Lengths of each sequence
        model.fit(X, lengths)  # Train the HMM model
        models[digit] = model
    return models

# Predict the digit based on highest HMM model score
def predict_digit(mfcc_features, models):
    max_score = float("-inf")
    best_digit = None
    for digit, model in models.items():
        score = model.score(mfcc_features)
        if score > max_score:
            max_score = score
            best_digit = digit
    return best_digit

# Load dataset and split it
train_files, val_files, test_files = split_dataset(dataset_path)

# Prepare training and testing data
train_data = prepare_data(train_files, dataset_path)
val_data = prepare_data(val_files, dataset_path)
test_data = prepare_data(test_files, dataset_path)

# Train models for each digit
models = train_hmm_models(train_data)

def evaluate_accuracy(test_data, models):
    correct = 0
    total = 0
    for digit, features in test_data.items():
        for mfcc_features in features:
            prediction = predict_digit(mfcc_features, models)
            # print(mfcc_features.shape)
            if prediction == digit:
                correct += 1
            total += 1
    accuracy = correct / total
    return accuracy * 100  # Return accuracy as a percentage

accuracy = evaluate_accuracy(test_data, models)
print(f"Test Accuracy: {accuracy:.2f}%")
accuracy = evaluate_accuracy(val_data, models)
print(f"Validation Accuracy: {accuracy:.2f}%")
accuracy = evaluate_accuracy(train_data, models)
print(f"Train Accuracy: {accuracy:.2f}%")

dataset_path_2 = '../../data/external/compressed'

# Test recordings and print predictions
for file_name in os.listdir(dataset_path_2):
    if file_name.endswith('.wav'):
        file_path = os.path.join(dataset_path_2, file_name)
        mfcc_features = extract_mfcc(file_path)
        predicted_digit = predict_digit(mfcc_features, models)
        print(f"File: {file_name}, Predicted Digit: {predicted_digit}")
