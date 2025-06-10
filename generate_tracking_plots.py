import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

def create_loss_accuracy_curves(df, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Loss curves
    ax1.plot(df['epoch'], df['train_loss'], 'o-', label='Training Loss', 
             color='#1f77b4', linewidth=2.5, markersize=8)
    ax1.plot(df['epoch'], df['val_loss'], 's-', label='Validation Loss', 
             color='#ff7f0e', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, len(df) + 0.5)
    
    # Plot 2: Accuracy curves
    ax2.plot(df['epoch'], df['train_acc'], 'o-', label='Training Accuracy', 
             color='#2ca02c', linewidth=2.5, markersize=8)
    ax2.plot(df['epoch'], df['val_acc'], 's-', label='Validation Accuracy', 
             color='#d62728', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, len(df) + 0.5)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def create_per_class_accuracy(df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each class
    ax.plot(df['epoch'], df['fastball_acc'], 'o-', label='Fastball', 
            color='#e74c3c', linewidth=2.5, markersize=10, marker='o')
    ax.plot(df['epoch'], df['breaking_acc'], 's-', label='Breaking Ball', 
            color='#3498db', linewidth=2.5, markersize=10, marker='s')
    ax.plot(df['epoch'], df['offspeed_acc'], '^-', label='Offspeed', 
            color='#2ecc71', linewidth=2.5, markersize=10, marker='^')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Per-Class Validation Accuracy', fontsize=16, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_ylim(-5, 105)
    
    ax.axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def create_class_performance_heatmap(df, save_path=None):
    class_data = df[['epoch', 'fastball_acc', 'breaking_acc', 'offspeed_acc']].set_index('epoch')
    class_data.columns = ['Fastball', 'Breaking Ball', 'Offspeed']
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    sns.heatmap(class_data.T, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=50, vmin=0, vmax=100,
                cbar_kws={'label': 'Accuracy (%)'}, 
                xticklabels=True, yticklabels=True, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Per-Class Accuracy Heatmap Across Epochs', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Pitch Type', fontsize=14)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def create_overfitting_analysis(df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    acc_gap = df['train_acc'] - df['val_acc']
    
    bars = ax.bar(df['epoch'], acc_gap, color=['#e74c3c' if g > 10 else '#3498db' for g in acc_gap],
                   alpha=0.7, edgecolor='black', linewidth=1)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Training - Validation Accuracy (%)', fontsize=14)
    ax.set_title('Overfitting Analysis: Training vs Validation Gap', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0.5, len(df) + 0.5)
    
    for bar, gap in zip(bars, acc_gap):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                f'{gap:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def create_final_performance_summary(df, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    final_epoch = df.iloc[-1]

    categories = ['Training', 'Validation']
    accuracies = [final_epoch['train_acc'], final_epoch['val_acc']]
    colors = ['#3498db', '#e74c3c']
    
    bars1 = ax1.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    ax1.set_title(f'Final Accuracy (Epoch {int(final_epoch["epoch"])})', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    classes = ['Fastball', 'Breaking Ball', 'Offspeed']
    class_accs = [final_epoch['fastball_acc'], final_epoch['breaking_acc'], final_epoch['offspeed_acc']]
    colors2 = ['#e74c3c', '#3498db', '#2ecc71']
    
    bars2 = ax2.bar(classes, class_accs, color=colors2, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_title('Final Per-Class Validation Accuracy', fontsize=16, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')

    for bar, acc in zip(bars2, class_accs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def create_learning_rate_visualization(df, initial_lr=1e-4, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = df['epoch'].values
    T_max = len(epochs)
    lrs = []
    
    for epoch in epochs:
        if epoch <= 3:
            lr = initial_lr * epoch / 3
        else:
            lr = initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - 3) / (T_max - 3)))
        lrs.append(lr)
    
    ax.plot(epochs, lrs, 'o-', color='#9b59b6', linewidth=2.5, markersize=8)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Learning Rate Schedule (Cosine Annealing with Warm-up)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xlim(0.5, len(df) + 0.5)
    
    ax.axvspan(0.5, 3.5, alpha=0.2, color='orange', label='Warm-up Phase')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def generate_all_charts(csv_file, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print("Generating visualizations...")
    
    print("  1. Creating loss and accuracy curves...")
    create_loss_accuracy_curves(df, f"{output_dir}/loss_accuracy_curves.png")
    
    print("  2. Creating per-class accuracy plot...")
    create_per_class_accuracy(df, f"{output_dir}/per_class_accuracy.png")
    
    print("  3. Creating class performance heatmap...")
    create_class_performance_heatmap(df, f"{output_dir}/class_performance_heatmap.png")
    
    print("  4. Creating overfitting analysis...")
    create_overfitting_analysis(df, f"{output_dir}/overfitting_analysis.png")
    
    print("  5. Creating final performance summary...")
    create_final_performance_summary(df, f"{output_dir}/final_performance_summary.png")
    
    print("  6. Creating learning rate visualization...")
    create_learning_rate_visualization(df, save_path=f"{output_dir}/learning_rate_schedule.png")
    
    print("\n Training Summary")
    best_val_epoch = df.loc[df['val_acc'].idxmax()]
    final_epoch = df.iloc[-1]
    
    print(f"Best validation accuracy: {best_val_epoch['val_acc']:.2f}% at epoch {int(best_val_epoch['epoch'])}")
    print(f"Final validation accuracy: {final_epoch['val_acc']:.2f}%")
    print(f"Final training accuracy: {final_epoch['train_acc']:.2f}%")
    print(f"Overfitting gap: {final_epoch['train_acc'] - final_epoch['val_acc']:.2f}%")
    
    print("\nFinal per-class accuracies:")
    print(f"  Fastball: {final_epoch['fastball_acc']:.2f}%")
    print(f"  Breaking Ball: {final_epoch['breaking_acc']:.2f}%")
    print(f"  Offspeed: {final_epoch['offspeed_acc']:.2f}%")
    
    print(f"\nAll charts saved to {output_dir}/")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training report visualizations')
    parser.add_argument('csv_file', help='Path to the CSV file with training data')
    parser.add_argument('--output_dir', default='./report_figures', 
                       help='Directory to save the generated charts')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    generate_all_charts(args.csv_file, args.output_dir)
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()