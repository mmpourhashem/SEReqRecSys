import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Configuration
dir_path = 'input_output_data'
output_formats = ['pdf']  # Add 'png' if needed: ['pdf', 'png']
dpi = 300

# Define methods with consistent styling
methods = [
    {'rmse': 'RMSE_ICF',           'mae': 'MAE_ICF',           'r2': 'R2_ICF',           'label': 'ICF',             'marker': 'D', 'color': 'blue'},
    {'rmse': 'RMSE_HCBCF',         'mae': 'MAE_HCBCF',         'r2': 'R2_HCBCF',         'label': 'HCBCF',           'marker': 's', 'color': 'red'},
    {'rmse': 'RMSE_HTS',           'mae': 'MAE_HTS',           'r2': 'R2_HTS',           'label': 'HTS',             'marker': 'o', 'color': 'orange'},
    {'rmse': 'RMSE_ProposedMethod','mae': 'MAE_ProposedMethod','r2': 'R2_ProposedMethod','label': 'Proposed Method', 'marker': 'v', 'color': 'green'}
]

# Create legend elements once (same for all plots)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker=m['marker'], color=m['color'], label=m['label'],
           markersize=7, linestyle='-', markerfacecolor=m['color'])
    for m in methods
]

# Dataset configurations: file, x-axis column, x-label
datasets = [
    {
        'file': 'sparsity_results.xlsx',
        'x_col': 'Sparsity',
        'xlabel': 'Sparsity'
    },
    {
        'file': 'item_coldStart_results.xlsx',
        'x_col': 'RatingsPerItem',
        'xlabel': 'Number of ratings for new requirements'
    },
    {
        'file': 'user_coldStart_results.xlsx',
        'x_col': 'RatingsPerUser',
        'xlabel': 'Number of ratings for new stakeholders'
    }
]

# Metric configurations
metrics = [
    {
        'name': 'RMSE',
        'y_col_key': 'rmse',
        'ylabel': 'RMSE',
        'higher_is_better': False
    },
    {
        'name': 'MAE',
        'y_col_key': 'mae',
        'ylabel': 'MAE',
        'higher_is_better': False
    },
    {
        'name': 'R2',
        'y_col_key': 'r2',
        'ylabel': r'$R^2$',
        'higher_is_better': True
    }
]

# Common plotting function
def plot_metric_comparison(df, x_col, x_label, metric_config, save_name):
    plt.figure(figsize=(8, 4))
    x_positions = np.arange(len(df))
    
    # Set x-axis
    plt.xticks(x_positions, df[x_col])
    plt.xlim(0, len(df) - 1)
    plt.xlabel(x_label)
    plt.ylabel(metric_config['ylabel'])
    plt.grid(True)

    # Plot each method
    for method in methods:
        metric_col = method[metric_config['y_col_key']]
        if metric_col not in df.columns:
            print(f"Warning: Column '{metric_col}' not found in data. Skipping.")
            continue
        plt.plot(
            x_positions, df[metric_col],
            label=f"{method['label']} {metric_config['ylabel']}",
            linestyle='-', marker=method['marker'], color=method['color']
        )

    # Shared legend below the plot
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.16),
               fancybox=False, shadow=False, ncol=4, frameon=False)
    plt.subplots_adjust(bottom=0.3)

    # Save in all formats
    for fmt in output_formats:
        plt.savefig(os.path.join(dir_path, f"{save_name}.{fmt}"), dpi=dpi, bbox_inches='tight')
    plt.close()

# === Main Loop: Iterate over datasets and metrics ===
for dataset in datasets:
    file_path = os.path.join(dir_path, dataset['file'])
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df = pd.read_excel(file_path)

    for metric in metrics:
        # Skip if no relevant columns exist
        relevant_cols = [m[metric['y_col_key']] for m in methods]
        if not all(col in df.columns for col in relevant_cols):
            print(f"Skipping {metric['name']} for {dataset['file']} – missing columns.")
            continue

        save_name = f"{metric['name']}_{dataset['file'].replace('.xlsx', '')}"
        plot_metric_comparison(
            df=df,
            x_col=dataset['x_col'],
            x_label=dataset['xlabel'],
            metric_config=metric,
            save_name=save_name
        )