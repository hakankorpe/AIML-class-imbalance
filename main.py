import sys
import os

# Add 'experiments' to path so we can import modules easily
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments'))

# --- POSTER 1 EXPERIMENTS ---
from experiments.run_breast_cancer import run as exp_breast
from experiments.run_breast_cancer_smote import run as exp_smote
from experiments.run_mnist_binary import run as exp_mnist
from experiments.run_fmnist_binary import run as exp_fmnist
from experiments.run_mnist_multi_digit import run as exp_mnist_multi

# --- POSTER 2 EXPERIMENTS ---
from experiments.run_mnist_smote import run as exp_mnist_smote
from experiments.run_mnist_aug import run as exp_mnist_aug

# --- NEW GAN PIPELINE IMPORTS ---
from experiments.train_gan import train as train_gan
from experiments.visualize_all import generate_comparison

# --- CONFIGURATION ---
# Set to True if you change hyperparameters and need to overwrite old models
FORCE_RETRAIN = False 

def run_gan_pipeline(digit):
    """
    Checks if a model exists for the given digit.
    If yes -> Skips training.
    If no -> Trains the model.
    Always -> Generates the visualization.
    """
    print(f"\n--- Processing GAN for Digit {digit} ---")
    
    model_filename = f"cgan_generator_{digit}.pth"
    
    # STEP 1: TRAIN (If needed)
    if os.path.exists(model_filename) and not FORCE_RETRAIN:
        print(f"[Skip Training] Found existing model '{model_filename}'.")
    else:
        if FORCE_RETRAIN:
            print(f"[Training] Force Retrain enabled for Digit {digit}...")
        else:
            print(f"[Training] Model not found. Training GAN for Digit {digit}...")
        
        # Calls the train function from experiments/train_gan.py
        train_gan(digit)

    # STEP 2: VISUALIZE
    print(f"[Visualizing] Generating comparison images for Digit {digit}...")
    # Calls the generator from experiments/visualize_all.py
    generate_comparison(digit)

if __name__ == "__main__":
    # ==========================================
    # PART 1: PREVIOUS EXPERIMENTS (POSTER 1)
    # ==========================================
    print("\n" + "="*40)
    print("      PART 1: BASELINE EXPERIMENTS")
    print("="*40)
    
    print("\n>>> Running Breast Cancer (Class Weights)...")
    exp_breast()

    print("\n>>> Running Breast Cancer (SMOTE)...")
    exp_smote()

    print("\n>>> Running MNIST Binary (Class Weights)...")
    exp_mnist()

    print("\n>>> Running Fashion-MNIST Binary (Class Weights)...")
    exp_fmnist()

    print("\n>>> Running MNIST multi-digit imbalance experiment...")
    exp_mnist_multi()

    # ==========================================
    # PART 2: ADVANCED METHODS (POSTER 2)
    # ==========================================
    print("\n" + "="*40)
    print("      PART 2: ADVANCED METHODS (SMOTE vs AUG vs GAN)")
    print("="*40)

    print("\n>>> Running MNIST (SMOTE - Phase 2)...")
    exp_mnist_smote()

    print("\n>>> Running MNIST (Augmentation - Phase 1)...")
    exp_mnist_aug()

    # ==========================================
    # PART 3: GAN PIPELINE (0 vs 8)
    # ==========================================
    print("\n>>> Running GAN Pipeline (Phase 3)...")
    
    # 1. Simple Topology (Digit 0)
    run_gan_pipeline(0)
    
    # 2. Complex Topology (Digit 8)
    run_gan_pipeline(8)

    print("\n" + "="*50)
    print(" ALL EXPERIMENTS COMPLETE.")
    print(" Check 'figures/' for your new comparison images.")
    print("="*50)