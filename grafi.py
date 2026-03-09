import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create the folder for graphs if it doesn't exist
output_folder = 'osic_graphs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Folder '{output_folder}' created successfully!")

# Set the style for graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the CSV file
df = pd.read_csv('OSIC_validation.csv')

# Create a column to identify problematic cases
df['has_fail'] = (df['max_gen_tech_status'] == 'FAIL') | \
                 (df['pc_ratio_tech_status'] == 'FAIL') | \
                 (df['tapering_tech_status'] == 'FAIL') | \
                 (df['branch_count_tech_status'] == 'FAIL') | \
                 (df['volume_tech_status'] == 'FAIL')

print("Generating graphs...")

# 1. PIE CHART: Distribution of RELIABLE vs UNRELIABLE
plt.figure(figsize=(10, 8))
status_counts = df['status'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for reliable, red for unreliable
wedges, texts, autotexts = plt.pie(status_counts.values, 
                                    labels=status_counts.index,
                                    autopct='%1.1f%%',
                                    colors=colors,
                                    startangle=90,
                                    explode=(0.05, 0.1))
plt.title('Distribution of Cases: RELIABLE vs UNRELIABLE', fontsize=16, fontweight='bold', pad=20)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)
plt.tight_layout()
plt.savefig(f'{output_folder}/1_status_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graph 1/8: Status distribution")

# 2. BAR CHART: Reasons for FAIL
plt.figure(figsize=(12, 8))
fail_reasons = ['max_generation', 'pc_ratio', 'tapering_ratio', 'branch_count', 'volume']
fail_counts = [
    (df['max_gen_tech_status'] == 'FAIL').sum(),
    (df['pc_ratio_tech_status'] == 'FAIL').sum(),
    (df['tapering_tech_status'] == 'FAIL').sum(),
    (df['branch_count_tech_status'] == 'FAIL').sum(),
    (df['volume_tech_status'] == 'FAIL').sum()
]
bars = plt.bar(fail_reasons, fail_counts, color=['#f39c12', '#9b59b6', '#3498db', '#e67e22', '#1abc9c'])
plt.title('FAIL Reasons by Parameter', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Number of FAILs', fontsize=12)
plt.xlabel('Parameter', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, count in zip(bars, fail_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(f'{output_folder}/2_fail_reasons.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graph 2/8: FAIL reasons")

# 3. SCATTER PLOT: Volume vs Branch Count
plt.figure(figsize=(12, 8))
colors_map = {'RELIABLE': '#2ecc71', 'UNRELIABLE': '#e74c3c'}
for status in df['status'].unique():
    mask = df['status'] == status
    plt.scatter(df.loc[mask, 'volume_ml'], 
                df.loc[mask, 'branch_count'],
                c=colors_map[status],
                label=status,
                alpha=0.7,
                s=150,
                edgecolors='black',
                linewidth=1)
plt.xlabel('Volume (ml)', fontsize=12)
plt.ylabel('Branch Count', fontsize=12)
plt.title('Volume vs Branch Count', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_folder}/3_volume_vs_branchcount.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graph 3/8: Volume vs Branch Count")

# 4. SCATTER PLOT: Volume vs PC Ratio (log scale)
plt.figure(figsize=(12, 8))
for status in df['status'].unique():
    mask = df['status'] == status
    plt.scatter(df.loc[mask, 'volume_ml'], 
                df.loc[mask, 'pc_ratio'],
                c=colors_map[status],
                label=status,
                alpha=0.7,
                s=150,
                edgecolors='black',
                linewidth=1)
plt.xlabel('Volume (ml)', fontsize=12)
plt.ylabel('PC Ratio (log scale)', fontsize=12)
plt.title('Volume vs PC Ratio', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig(f'{output_folder}/4_volume_vs_pcratio.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graph 4/8: Volume vs PC Ratio")

# 5. SCATTER PLOT: Branch Count vs Max Generation
plt.figure(figsize=(12, 8))
for status in df['status'].unique():
    mask = df['status'] == status
    plt.scatter(df.loc[mask, 'branch_count'], 
                df.loc[mask, 'max_generation'],
                c=colors_map[status],
                label=status,
                alpha=0.7,
                s=150,
                edgecolors='black',
                linewidth=1)
plt.xlabel('Branch Count', fontsize=12)
plt.ylabel('Max Generation', fontsize=12)
plt.title('Branch Count vs Max Generation', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_folder}/5_branchcount_vs_maxgen.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graph 5/8: Branch Count vs Max Generation")

# 6. BOX PLOT: Volume by Status
plt.figure(figsize=(10, 8))
df.boxplot(column='volume_ml', by='status', grid=False, patch_artist=True,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='red', linewidth=2))
plt.title('Volume Distribution by Status', fontsize=16, fontweight='bold', pad=20)
plt.suptitle('')  # Remove automatic title
plt.xlabel('Status', fontsize=12)
plt.ylabel('Volume (ml)', fontsize=12)
plt.yscale('log')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_folder}/6_boxplot_volume.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graph 6/8: Box Plot Volume")

# 7. CORRELATION MATRIX
plt.figure(figsize=(10, 8))
numeric_cols = ['volume_ml', 'branch_count', 'max_generation', 'pc_ratio', 'tapering_ratio']
correlation_matrix = df[numeric_cols].corr()

# Create heatmap
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 12})
plt.title('Correlation Matrix of Parameters', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_folder}/7_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Graph 7/8: Correlation Matrix")

# 8. BAR CHART: Critical parameters for UNRELIABLE cases
plt.figure(figsize=(14, 8))
unreliable_df = df[df['status'] == 'UNRELIABLE']

if len(unreliable_df) > 0:
    x = np.arange(len(unreliable_df))
    width = 0.15

    # Create multiple bars for each unreliable case
    metrics = ['volume_ml', 'branch_count', 'max_generation', 'pc_ratio', 'tapering_ratio']
    colors_metrics = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#f1c40f']
    metric_labels = ['Volume', 'Branch Count', 'Max Generation', 'PC Ratio', 'Tapering Ratio']

    for i, (metric, color, label) in enumerate(zip(metrics, colors_metrics, metric_labels)):
        # Normalize values for visualization
        min_val = unreliable_df[metric].min()
        max_val = unreliable_df[metric].max()
        if max_val > min_val:  # Avoid division by zero
            normalized_values = (unreliable_df[metric] - min_val) / (max_val - min_val)
        else:
            normalized_values = [0.5] * len(unreliable_df)  # Default value if all equal
        
        plt.bar(x + i*width, normalized_values, width, label=label, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

    plt.xlabel('UNRELIABLE Case ID', fontsize=12)
    plt.ylabel('Normalized Values', fontsize=12)
    plt.title('Parameter Comparison for UNRELIABLE Cases', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x + width * 2, [f'Case {i+1}' for i in range(len(unreliable_df))], rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/8_unreliable_cases_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Graph 8/8: UNRELIABLE cases comparison")
else:
    print("⚠ No UNRELIABLE cases found, graph 8 not generated")
plt.close()

# BONUS GRAPH: Distribution of parameters
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

parameters = ['volume_ml', 'branch_count', 'max_generation', 'pc_ratio', 'tapering_ratio']
titles = ['Volume (ml)', 'Branch Count', 'Max Generation', 'PC Ratio', 'Tapering Ratio']

for i, (param, title) in enumerate(zip(parameters, titles)):
    for status in df['status'].unique():
        mask = df['status'] == status
        axes[i].hist(df.loc[mask, param], alpha=0.7, label=status, bins=15, edgecolor='black')
    axes[i].set_xlabel(title, fontsize=11)
    axes[i].set_ylabel('Frequency', fontsize=11)
    axes[i].set_title(f'Distribution of {title}', fontsize=13, fontweight='bold')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Remove the empty last subplot
fig.delaxes(axes[5])

plt.suptitle('Distribution of Parameters by Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_folder}/9_parameter_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Bonus Graph: Parameter distribution")

# Print statistics and save to a txt file
print("\n" + "="*60)
print("GRAPH GENERATION COMPLETED!")
print("="*60)
print(f"All graphs have been saved in the folder: '{output_folder}'")

# Save statistics to a file
with open(f'{output_folder}/statistics.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("SUMMARY STATISTICS - OSIC VALIDATION\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total cases: {len(df)}\n")
    f.write(f"RELIABLE cases: {(df['status'] == 'RELIABLE').sum()} ({((df['status'] == 'RELIABLE').sum()/len(df)*100):.1f}%)\n")
    f.write(f"UNRELIABLE cases: {(df['status'] == 'UNRELIABLE').sum()} ({((df['status'] == 'UNRELIABLE').sum()/len(df)*100):.1f}%)\n\n")
    f.write("FAIL details by parameter:\n")
    f.write(f"- max_generation: {(df['max_gen_tech_status'] == 'FAIL').sum()} FAIL\n")
    f.write(f"- pc_ratio: {(df['pc_ratio_tech_status'] == 'FAIL').sum()} FAIL\n")
    f.write(f"- tapering_ratio: {(df['tapering_tech_status'] == 'FAIL').sum()} FAIL\n")
    f.write(f"- branch_count: {(df['branch_count_tech_status'] == 'FAIL').sum()} FAIL\n")
    f.write(f"- volume: {(df['volume_tech_status'] == 'FAIL').sum()} FAIL\n\n")
    
    if len(unreliable_df) > 0:
        f.write("Descriptive statistics for UNRELIABLE cases:\n")
        f.write(unreliable_df[['volume_ml', 'branch_count', 'max_generation', 'pc_ratio', 'tapering_ratio']].describe().to_string())
        f.write("\n\nUNRELIABLE cases details:\n")
        for idx, row in unreliable_df.iterrows():
            f.write(f"\n{row['case']}:\n")
            f.write(f"  - Volume: {row['volume_ml']} ml\n")
            f.write(f"  - Branch Count: {row['branch_count']}\n")
            f.write(f"  - Max Generation: {row['max_generation']}\n")
            f.write(f"  - PC Ratio: {row['pc_ratio']}\n")
            f.write(f"  - Tapering Ratio: {row['tapering_ratio']}\n")

print(f"\n📊 Statistics saved in: {output_folder}/statistics.txt")
print("\nList of generated graphs:")
for file in sorted(os.listdir(output_folder)):
    if file.endswith('.png'):
        print(f"  - {file}")