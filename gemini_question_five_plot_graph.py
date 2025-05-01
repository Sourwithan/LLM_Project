import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import re

def load_json_files(letter):
    """Load multiple JSON files matching the pattern gemini_Q5_{letter}*."""
    pattern = f"gemini_Q5_{letter}*"
    files = glob.glob(pattern)
    data_list = []
    
    print(f"Looking for files matching pattern: {pattern}")
    
    if not files:
        print(f"No files found matching pattern {pattern}")
        return data_list
    
    print(f"Found {len(files)} files:")
    for file_path in files:
        print(f"  - {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Add filename as metadata
                data['file_name'] = os.path.basename(file_path)
                data_list.append(data)
        except json.JSONDecodeError:
            print(f"  Error: Could not parse {file_path} as JSON")
        except Exception as e:
            print(f"  Error loading {file_path}: {str(e)}")
    
    return data_list

def extract_filename_info(filename):
    """Extract information from filename like 'gemini_Q5_P-5W-10L'."""
    # Try to extract the suffix pattern like '5W-10L'
    parts = filename.split('-')
    if len(parts) >= 3:
        # Get the last two parts which should be like '5W' and '10L'
        try:
            suffix = f"{parts[-2]}-{parts[-1]}"
            return suffix
        except:
            pass
    
    # If we can't extract a clean pattern, return the original filename
    return filename

def get_sorting_key(display_name):
    """Extract numerical values from display names like '5W-10L' for sorting."""
    # Extract numbers using regex
    numbers = re.findall(r'(\d+)[WL]', display_name)
    
    if len(numbers) >= 2:
        try:
            first_num = int(numbers[0])
            second_num = int(numbers[1])
            # Return as tuple for sorting
            return (first_num, second_num)
        except ValueError:
            pass
    
    # Default sorting key if we can't extract numbers
    return (0, 0)

def extract_metrics(data_list):
    """Extract key metrics from the JSON data for comparison."""
    metrics = []
    
    for data in data_list:
        file_name = data['file_name']
        avg_metrics = data['avg_metrics']
        
        # Extract the suffix for better labeling in plots
        display_name = extract_filename_info(file_name)
        
        # Get sorting key
        sort_key = get_sorting_key(display_name)
        
        metrics.append({
            'file_name': file_name,
            'display_name': display_name,
            'success_rate': avg_metrics['success_rate'],
            'criteria_accuracy': avg_metrics['avg_criteria_accuracy'],
            'count_accuracy': avg_metrics['avg_count_accuracy'],
            'overall_accuracy': avg_metrics['avg_overall_accuracy'],
            'total_unique_words': avg_metrics['total_unique_words'],
            'total_duration': avg_metrics['total_duration_seconds'],
            'std_dev_criteria': avg_metrics.get('std_dev_criteria_accuracy', 0),
            'std_dev_count': avg_metrics.get('std_dev_count_accuracy', 0),
            'std_dev_overall': avg_metrics.get('std_dev_overall_accuracy', 0),
            'sort_key_first': sort_key[0],  # Store as separate columns for easier sorting
            'sort_key_second': sort_key[1]
        })
    
    # Create DataFrame
    df = pd.DataFrame(metrics)
    
    # Sort by the two numerical keys
    df = df.sort_values(by=['sort_key_first', 'sort_key_second'], ascending=True)
    
    return df

def combine_word_frequencies(data_list):
    """Combine word frequencies from all files."""
    all_words = Counter()
    
    for data in data_list:
        for word, count in data['avg_metrics']['most_common_words']:
            all_words[word] += count
    
    return all_words

def plot_performance_metrics(metrics_df, letter):
    """Plot detailed performance and accuracy metrics."""
    if metrics_df.empty:
        print("No data to plot.")
        return
    
    # Extract x labels
    x_labels = metrics_df['display_name'].tolist()
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Performance Metrics for {letter}-Words', fontsize=16)
    
    # 1. Success Rate
    ax1 = axes[0, 0]
    ax1.bar(range(len(x_labels)), metrics_df['success_rate'], color='green')
    ax1.set_title('Success Rate')
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(metrics_df['success_rate']):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # 2. Accuracy Metrics Comparison
    ax2 = axes[0, 1]
    width = 0.25
    x = np.arange(len(x_labels))
    
    ax2.bar(x - width, metrics_df['criteria_accuracy'], width, label='Criteria Accuracy', color='skyblue')
    ax2.bar(x, metrics_df['count_accuracy'], width, label='Count Accuracy', color='orange')
    ax2.bar(x + width, metrics_df['overall_accuracy'], width, label='Overall Accuracy', color='green')
    
    ax2.set_title('Accuracy Metrics')
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.legend()
    
    # 3. Standard Deviations of Accuracy
    ax3 = axes[1, 0]
    width = 0.25
    x = np.arange(len(x_labels))
    
    ax3.bar(x - width, metrics_df['std_dev_criteria'], width, label='Criteria Std Dev', color='skyblue')
    ax3.bar(x, metrics_df['std_dev_count'], width, label='Count Std Dev', color='orange')
    ax3.bar(x + width, metrics_df['std_dev_overall'], width, label='Overall Std Dev', color='green')
    
    ax3.set_title('Standard Deviation of Accuracy')
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, rotation=45)
    ax3.legend()
    
    # 4. Processing Efficiency (Unique Words / Duration)
    ax4 = axes[1, 1]
    efficiency = metrics_df['total_unique_words'] / np.maximum(metrics_df['total_duration'], 0.001)  # Avoid division by zero
    ax4.bar(range(len(x_labels)), efficiency, color='purple')
    ax4.set_title('Processing Efficiency (Unique Words / Second)')
    ax4.set_xticks(range(len(x_labels)))
    ax4.set_xticklabels(x_labels, rotation=45)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust for the suptitle
    plt.savefig(f'performance_metrics_{letter}.png')
    plt.close()
    print(f"Performance metrics plot saved as 'performance_metrics_{letter}.png'")

def plot_accuracy_comparison(metrics_df, letter):
    """Plot a detailed comparison of accuracy metrics."""
    if metrics_df.empty:
        print("No data to plot.")
        return
    
    # Create a radar chart (polar plot) for each file
    categories = ['Success Rate', 'Criteria Accuracy', 'Count Accuracy', 'Overall Accuracy']
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create a new figure for radar chart
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the first axis to be on top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Plot each file's data
    for _, row in metrics_df.iterrows():
        # Add the values for each category
        values = [row['success_rate'], row['criteria_accuracy'], 
                  row['count_accuracy'], row['overall_accuracy']]
        values += values[:1]  # Close the loop
        
        # Plot the line and fill area
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=row['display_name'])
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Accuracy Comparison for {letter}-Words', size=15)
    
    plt.tight_layout()
    plt.savefig(f'accuracy_comparison_{letter}.png')
    plt.close()
    print(f"Accuracy comparison plot saved as 'accuracy_comparison_{letter}.png'")

def main():
    letter = 'C'
    
    # Load all JSON files matching the pattern
    data_list = load_json_files(letter)
    
    if not data_list:
        return
    
    # Extract metrics and sort data
    metrics_df = extract_metrics(data_list)
    
    # Plot detailed performance and accuracy metrics
    plot_performance_metrics(metrics_df, letter)
    
    # Plot accuracy comparison radar chart
    plot_accuracy_comparison(metrics_df, letter)
    
    # Generate summary report
    print("\nPerformance Summary:")
    print(f"Number of files analyzed: {len(data_list)}")
    print(f"Average success rate: {metrics_df['success_rate'].mean():.2f}")
    print(f"Average overall accuracy: {metrics_df['overall_accuracy'].mean():.2f}")
    
    # Save detailed summary to file
    with open(f'performance_summary_{letter}_words.txt', 'w') as f:
        f.write(f"Performance Analysis of {letter}-Words from {len(data_list)} files\n")
        f.write("=============================================\n\n")
        f.write("ACCURACY METRICS:\n")
        f.write(f"Average success rate: {metrics_df['success_rate'].mean():.4f}\n")
        f.write(f"Average criteria accuracy: {metrics_df['criteria_accuracy'].mean():.4f}\n")
        f.write(f"Average count accuracy: {metrics_df['count_accuracy'].mean():.4f}\n")
        f.write(f"Average overall accuracy: {metrics_df['overall_accuracy'].mean():.4f}\n\n")
        
        f.write("PERFORMANCE BY FILE (sorted by word dimensions):\n")
        for _, row in metrics_df.iterrows():
            f.write(f"  {row['display_name']} ({row['file_name']}):\n")
            f.write(f"    Success Rate: {row['success_rate']:.4f}\n")
            f.write(f"    Criteria Accuracy: {row['criteria_accuracy']:.4f}\n")
            f.write(f"    Count Accuracy: {row['count_accuracy']:.4f}\n")
            f.write(f"    Overall Accuracy: {row['overall_accuracy']:.4f}\n")
            f.write(f"    Processing Time: {row['total_duration']:.2f} seconds\n\n")
    
    print(f"Performance summary saved to 'performance_summary_{letter}_words.txt'")

if __name__ == "__main__":
    main()