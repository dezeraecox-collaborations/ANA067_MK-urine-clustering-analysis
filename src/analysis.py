import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from loguru import logger

logger.info('Import OK')

input_folder = 'experimental_data/big table.xlsx'
output_folder = 'results/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    

# Read in data
raw_data = pd.read_excel(input_folder)
raw_data.drop(0, axis=0, inplace=True)

# Clean column names and filter anything as needed
tht_cols = raw_data.columns[1:4]
tht = raw_data[tht_cols].mean(axis=1).tolist()
ascp_cols = raw_data.columns[4:7]
ascp = raw_data[ascp_cols].mean(axis=1).tolist()
dep_cols = raw_data.columns[7:10]
dep = raw_data[dep_cols].mean(axis=1).tolist()
other_cols = raw_data.columns[10:]

clean_data = raw_data[['Unnamed: 0']].copy()
clean_data.columns = ['Sample #']
clean_data['ThT'] = tht
clean_data['ASCP'] = ascp
clean_data['Depanzine'] = dep

for col in other_cols:
    clean_data[col] = raw_data[col]

# Save cleaned copy of the data
clean_data.to_csv(f'{output_folder}clean_data.csv')

# ===================Visualise the pairplot===================
melted = pd.melt(clean_data, id_vars='Sample #')

sns.pairplot(data=clean_data, size=50)

## Define a colourscheme

clean_data['Sample #'].unique()

## Generate figure


# Create a PairGrid for custom plots

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

import matplotlib
font = {'family' : 'arial',
'weight' : 'normal',
'size'   : 10 }
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'

# Assuming `clean_data` is already defined and contains your dataset

# Select numeric columns for the pairplot (excluding 'Sample #' or other non-numeric columns)
numeric_cols = [col for col in clean_data.columns if 'Sample #' not in col]
data = clean_data[numeric_cols]

# Create a custom grid for the plot
fig, axes = plt.subplots(len(data.columns), len(data.columns), figsize=(12, 12))

# Function to calculate and annotate Pearson's correlation on the heatmap
def annotate_heatmap(x, y, ax):
    r, p = pearsonr(x, y)
    ax.clear()  # Clear the axis to avoid overlapping plots
    heatmap = sns.heatmap(
        np.array([[r]]), 
        annot=True, 
        fmt=".2f",  # Format for correlation coefficient
        cbar=False,  # Disable colorbar here
        square=True, 
        vmin=-1, 
        vmax=1, 
        cmap='coolwarm', 
        xticklabels=False, 
        yticklabels=False, 
        ax=ax
    )
    # Annotate p-value below the heatmap
    ax.annotate(f'p={p:.2e}', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=8)
    return heatmap

# Function to create a plain regression plot
def regplot_plain(x, y, ax):
    ax.clear()  # Clear the axis to avoid overlapping plots
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'s': 10, 'color': 'black'}, line_kws={'color': 'grey'})

# Loop through the grid and plot
for i, col1 in enumerate(data.columns):
    for j, col2 in enumerate(data.columns):
        ax = axes[i, j]
        if i > j:  # Lower triangle
            regplot_plain(data[col2], data[col1], ax)
            # Remove y-axis labels for all but the leftmost plots
            if j != 0:
                ax.set_ylabel('')
            # Remove x-axis labels for all but the bottom row
            if i != len(data.columns) - 1:
                ax.set_xlabel('')
        elif i < j:  # Upper triangle
            annotate_heatmap(data[col2], data[col1], ax)
            # Add rotated y-axis labels on the right-hand side
            if j == len(data.columns) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(col1, fontsize=10, rotation=270, labelpad=15)
        else:  # Diagonal
            ax.clear()
            sns.histplot(data[col1], ax=ax, kde=True, color='black', line_kws={'color': 'grey'})
            ax.set_ylabel('')
            ax.set_xlabel('')
        # Remove only upper and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Remove ticks for non-edge subplots
        if i != len(data.columns) - 1:
            ax.set_xticks([])
        if j != 0:
            ax.set_yticks([])
        # Add labels to the top row and rightmost column
        if i == 0:
            ax.set_title(col2, fontsize=10)  # Top labels
        if j == len(data.columns) - 1:
            ax.set_ylabel(col1, fontsize=10, rotation=270, labelpad=15)  # Right labels

# Add missing labels for the top-leftmost and bottom-rightmost plots
axes[0, 0].set_ylabel(data.columns[0], fontsize=10)
axes[-1, -1].yaxis.set_label_position("right")  # Move y-axis label to the right
axes[-1, -1].set_ylabel(data.columns[-1], fontsize=10, rotation=270, labelpad=15)
axes[-1, -1].set_xlabel(data.columns[-1], fontsize=10)

# Adjust layout to add a tiny bit of whitespace
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(f'{output_folder}correlation-pairplot.svg')

# Create a standalone colorbar legend
fig_colorbar, ax_colorbar = plt.subplots(figsize=(2, 5))
sns.heatmap(
    np.array([[0]]),  # Dummy data for the colorbar
    cbar=True,
    square=True,
    vmin=-1,
    vmax=1,
    cmap='coolwarm',
    cbar_kws={'orientation': 'vertical', 'label': 'Correlation Coefficient'},
    ax=ax_colorbar
)
ax_colorbar.remove()  # Remove the dummy heatmap, leaving only the colorbar

# Show the plots
plt.savefig(f'{output_folder}cbar.svg')
plt.show()
fig_colorbar.show()

# =========================Perform PCA=========================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns

# Truncate the magma colormap to remove the last 10% of colors
def truncate_colormap(cmap, minval=0.0, maxval=0.9, n=256):
    new_cmap = LinearSegmentedColormap.from_list(
        f'truncated({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

# Create a truncated magma colormap
original_cmap = plt.cm.magma
truncated_cmap = truncate_colormap(original_cmap, maxval=0.9)

# Select numeric columns for PCA (excluding 'Sample #' or other non-numeric columns)
numeric_cols = [col for col in clean_data.columns if 'Sample #' not in col]
data_for_pca = clean_data[numeric_cols]


# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_pca)

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
pca_result = pca.fit_transform(scaled_data)

# Add PCA results to the original dataframe
clean_data['PCA1'] = pca_result[:, 0]
clean_data['PCA2'] = pca_result[:, 1]

# Visualize the PCA results
plt.figure()
sns.scatterplot(
    x='PCA1', 
    y='PCA2', 
    data=clean_data, 
    color='black',
    s=100
)
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.savefig(f'{output_folder}PCA.svg')
plt.savefig(f'{output_folder}PCA.png')
plt.show()

