"""
Script to download and prepare Wine Quality dataset
"""

import pandas as pd
import numpy as np
from urllib import request
import os

# Wine Quality dataset URLs
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Download datasets
print("Downloading Wine Quality dataset...")

try:
    # Download red wine data
    print("Downloading red wine data...")
    red_df = pd.read_csv(red_wine_url, sep=';')
    red_df['wine_type'] = 'red'
    
    # Download white wine data
    print("Downloading white wine data...")
    white_df = pd.read_csv(white_wine_url, sep=';')
    white_df['wine_type'] = 'white'
    
    # Combine datasets
    df = pd.concat([red_df, white_df], ignore_index=True)
    
    # For multi-class classification, keep the quality column as is
    # The quality is already categorical (3-9)
    
    # Rearrange columns - move target to the end
    target_col = 'quality'
    feature_cols = [col for col in df.columns if col != target_col]
    df = df[feature_cols + [target_col]]
    
    # Remove any rows with missing values
    df = df.dropna()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Samples: {df.shape[0]}")
    print(f"\nColumn names:")
    print(df.columns.tolist())
    print(f"\nTarget distribution:")
    print(df[target_col].value_counts().sort_index())
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    output_file = 'data/wine_quality.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Dataset saved to '{output_file}'")
    print(f"✓ Dataset has {df.shape[1]} columns and {df.shape[0]} rows")
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
    raise

print("\n" + "="*60)
print("Dataset preparation complete!")
print("="*60)
