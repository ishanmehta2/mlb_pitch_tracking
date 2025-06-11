import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
    

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def parse_log_file(filename):
    df = pd.read_csv(filename, header=None)
    df = df.iloc[:, 1:]  # Skip first column
    
    column_names = ['epoch', 'batch', 'total_samples', 'split', 
                    'loss', 'accuracy', 'fastball_acc', 'breaking_acc', 'offspeed_acc']
    

    df.columns = column_names[:len(df.columns)]

    numeric_columns = ['epoch', 'batch', 'total_samples', 'loss', 'accuracy', 
                      'fastball_acc', 'breaking_acc', 'offspeed_acc']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing per-class accuracies with NaN
    for col in ['fastball_acc', 'breaking_acc', 'offspeed_acc']:
        if col not in df.columns:
            df[col] = np.nan
    
    df = df[df['split'].isin(['train', 'val', 'test'])]
    
    return df

def plot_training_progress(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Progress Analysis', fontsize=16, y=0.995)
    
    ax1 = axes[0, 0]
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    ax1.plot(train_df['total_samples'], train_df['loss'], 
             label='Train Loss', color='blue', alpha=0.7, linewidth=2)
    
    ax1.scatter(val_df['total_samples'], val_df['loss'], 
                label='Val Loss', color='red', s=100, marker='o', zorder=5)
    
    ax1.set_xlabel('Total Samples Seen')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]

    ax2.plot(train_df['total_samples'], train_df['accuracy'], 
             label='Train Acc', color='green', alpha=0.7, linewidth=2)
    
    ax2.scatter(val_df['total_samples'], val_df['accuracy'], 
                label='Val Acc', color='orange', s=100, marker='o', zorder=5)
    
    ax2.set_xlabel('Total Samples Seen')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Per-class validation accuracies
    ax3 = axes[1, 0]
    
    if not val_df.empty and val_df['fastball_acc'].notna().any():
        epochs = val_df['epoch'].values
        
        ax3.plot(epochs, val_df['fastball_acc'], 
                 label='Fastball', marker='o', linewidth=2, markersize=8)
        ax3.plot(epochs, val_df['breaking_acc'], 
                 label='Breaking', marker='s', linewidth=2, markersize=8)
        ax3.plot(epochs, val_df['offspeed_acc'], 
                 label='Offspeed', marker='^', linewidth=2, markersize=8)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Per-Class Validation Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(epochs)
    
    ax4 = axes[1, 1]
    
    epoch_stats = []
    for epoch in train_df['epoch'].unique():
        epoch_data = train_df[train_df['epoch'] == epoch]
        epoch_stats.append({
            'epoch': epoch,
            'mean_loss': epoch_data['loss'].mean(),
            'std_loss': epoch_data['loss'].std(),
            'mean_acc': epoch_data['accuracy'].mean(),
            'std_acc': epoch_data['accuracy'].std(),
            'min_acc': epoch_data['accuracy'].min(),
            'max_acc': epoch_data['accuracy'].max()
        })
    
    epoch_stats_df = pd.DataFrame(epoch_stats)
    
    x = epoch_stats_df['epoch']
    y = epoch_stats_df['mean_acc']
    yerr = epoch_stats_df['std_acc']
    
    bars = ax4.bar(x, y, yerr=yerr, capsize=5, alpha=0.7, 
                    color='skyblue', edgecolor='navy', linewidth=1)
    
    for i, (epoch, row) in enumerate(epoch_stats_df.iterrows()):
        ax4.plot([row['epoch'], row['epoch']], 
                 [row['min_acc'], row['max_acc']], 
                 'k-', alpha=0.5, linewidth=2)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training Accuracy (%)')
    ax4.set_title('Training Accuracy Distribution by Epoch')
    ax4.set_xticks(x)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_learning_curves(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Learning Curves', fontsize=16)
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    train_epoch_end = train_df.groupby('epoch').last().reset_index()
    

    epochs = train_epoch_end['epoch'].values
    train_losses = train_epoch_end['loss'].values
    val_losses = val_df['loss'].values[:len(epochs)]
    
    ax1.plot(epochs, train_losses, 'o-', label='Train', linewidth=2, markersize=8)
    ax1.plot(val_df['epoch'].values, val_losses, 's-', label='Validation', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    train_accs = train_epoch_end['accuracy'].values
    val_accs = val_df['accuracy'].values[:len(epochs)]
    
    ax2.plot(epochs, train_accs, 'o-', label='Train', linewidth=2, markersize=8)
    ax2.plot(val_df['epoch'].values, val_accs, 's-', label='Validation', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    plt.tight_layout()
    return fig

def plot_class_performance_heatmap(df):
    val_df = df[df['split'] == 'val'].copy()
    
    if val_df.empty or val_df['fastball_acc'].isna().all():
        print("No per-class validation data available for heatmap")
        return None

    class_data = val_df[['epoch', 'fastball_acc', 'breaking_acc', 'offspeed_acc']].set_index('epoch')
    class_data.columns = ['Fastball', 'Breaking', 'Offspeed']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(class_data.T, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Accuracy (%)'}, 
                xticklabels=True, yticklabels=True, ax=ax)
    
    ax.set_title('Per-Class Accuracy Heatmap Across Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pitch Type')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize training progress from CSV log file')
    parser.add_argument('log_file', help='Path to the CSV log file')
    parser.add_argument('--save_dir', default='.', help='Directory to save plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()
    
    print(f"Reading CSV log file: {args.log_file}")
    df = parse_log_file(args.log_file)
    print(f"Loaded {len(df)} log entries")
    print(f"Columns found: {list(df.columns)}")
    print(f"Epochs: {df['epoch'].unique()}")
    
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\nCreating training progress plots...")
    fig1 = plot_training_progress(df)
    fig1.savefig(f"{args.save_dir}/training_progress.png", dpi=300, bbox_inches='tight')
    
    print("Creating learning curves...")
    fig2 = plot_learning_curves(df)
    fig2.savefig(f"{args.save_dir}/learning_curves.png", dpi=300, bbox_inches='tight')
    
    print("Creating class performance heatmap...")
    fig3 = plot_class_performance_heatmap(df)
    if fig3:
        fig3.savefig(f"{args.save_dir}/class_performance_heatmap.png", dpi=300, bbox_inches='tight')
    
    print("\n=== Training Summary ===")
    val_df = df[df['split'] == 'val']
    if not val_df.empty:
        best_epoch = val_df.loc[val_df['accuracy'].idxmax()]
        print(f"Best validation accuracy: {best_epoch['accuracy']:.2f}% at epoch {best_epoch['epoch']}")
        print(f"Final validation accuracy: {val_df.iloc[-1]['accuracy']:.2f}%")
        
        if not val_df['fastball_acc'].isna().all():
            final_val = val_df.iloc[-1]
            print(f"\nFinal per-class accuracies:")
            print(f"  Fastball: {final_val['fastball_acc']:.2f}%")
            print(f"  Breaking: {final_val['breaking_acc']:.2f}%")
            print(f"  Offspeed: {final_val['offspeed_acc']:.2f}%")
    
    train_df = df[df['split'] == 'train']
    if not train_df.empty:
        print(f"\nTraining statistics:")
        print(f"  Final training accuracy: {train_df.iloc[-1]['accuracy']:.2f}%")
        print(f"  Average training loss: {train_df['loss'].mean():.4f}")
    
    if args.show:
        plt.show()
    
    print(f"\nPlots saved to {args.save_dir}/")

if __name__ == "__main__":
    main()
