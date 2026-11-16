import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from ddpm.config import Config

def plot_training_curve():
    csv_path = os.path.join(Config.OUTPUT_DIR, "ddpm_training_history.csv")
    if not os.path.exists(csv_path):
        print("Training history not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss (MSE)')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Noise Prediction Loss')
    plt.title('Conditional DDPM Training Curve')
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(Config.ROOT_DIR, "results", "figures", "ddpm", "training_curve.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved training curve to {out_path}")

if __name__ == "__main__":
    plot_training_curve()
