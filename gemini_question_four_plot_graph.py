import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from datetime import datetime

def extract_wl_pattern(filename):
    """Extract word count and word length from filename."""
    # Pattern to match 'gemini_Q4_5W.json' or 'gemini_Q4_5W-5L.json'
    match_with_length = re.search(r'gemini_Q4_(\d+)W-(\d+)L', filename)
    match_without_length = re.search(r'gemini_Q4_(\d+)W(?!-)', filename)
    
    if match_with_length:
        return int(match_with_length.group(1)), int(match_with_length.group(2))
    elif match_without_length:
        return int(match_without_length.group(1)), None
    
    return None, None

def read_json_file(filepath):
    """Read a JSON file and return the data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def collect_metrics(file_pattern="gemini_Q4_*.json"):
    """Collect metrics from all matching files."""
    results = []
    word_data = {}
    
    for filepath in glob(file_pattern):
        word_count, word_length = extract_wl_pattern(filepath)
        if word_count is None:
            print(f"Could not extract pattern from: {filepath}")
            continue
            
        data = read_json_file(filepath)
        
        # Get the metrics
        metrics = data["avg_metrics"]
        
        # For files without length constraint in the filename,
        # check if there's a word_length in the metrics
        actual_word_length = word_length if word_length is not None else metrics.get("word_length", "N/A")
        
        # Check if length_constraint is relevant
        has_length_constraint = word_length is not None or "word_length" in metrics
        
        result = {
            "word_count": word_count,
            "word_length": actual_word_length,
            "has_length_constraint": has_length_constraint,
            "success_rate": metrics["success_rate"],
            "english_words_accuracy": metrics["avg_accuracy"]["english_words"],
            "vowel_constraint_accuracy": metrics["avg_accuracy"]["vowel_constraint"],
            "length_constraint_accuracy": metrics["avg_accuracy"]["length_constraint"] if has_length_constraint else None,
            "fully_valid_accuracy": metrics["avg_accuracy"]["fully_valid"],
            "count_accuracy": metrics["avg_accuracy"]["correct_count"],
            "total_words": metrics["total_words"],
            "unique_words": metrics["unique_words"],
            "uniqueness_ratio": metrics["uniqueness_ratio"],
            "total_duration_seconds": metrics["total_duration_seconds"]
        }
        
        # Create a config key for storing word data
        config_key = f"{word_count}W"
        if has_length_constraint:
            config_key += f"-{actual_word_length}L"
            
        # Store information about common words
        word_data[config_key] = metrics["most_common_words"]
        
        results.append(result)
    
    return pd.DataFrame(results), word_data

def plot_accuracy_metrics(df, timestamp):
    """Create visualizations for accuracy metrics."""
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Create config labels for x-axis
    df["config_label"] = df.apply(
        lambda x: f"{x['word_count']}W-{x['word_length']}L" if x["has_length_constraint"] 
        else f"{x['word_count']}W", axis=1
    )
    
    # Sort by word count and word length for better visualization
    df = df.sort_values(by=["word_count", "word_length"])
    
    # 1. Create grouped bar chart for accuracy metrics
    categories = df["config_label"].tolist()
    
    # Determine which metrics to include based on whether length constraint exists
    if df["has_length_constraint"].all():
        metrics = ["english_words_accuracy", "vowel_constraint_accuracy", 
                  "length_constraint_accuracy", "fully_valid_accuracy"]
        labels = ["English Words", "Valid Vowels", "Valid Length", "Fully Valid"]
    else:
        # Skip length constraint metric for some or all files
        metrics = ["english_words_accuracy", "vowel_constraint_accuracy", "fully_valid_accuracy"]
        labels = ["English Words", "Valid Vowels", "Fully Valid"]
    
    x = np.arange(len(categories))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        if metric == "length_constraint_accuracy" and not df["has_length_constraint"].all():
            # Skip for files without length constraint
            values = [row[metric] if row["has_length_constraint"] else 0 for _, row in df.iterrows()]
        else:
            values = df[metric].tolist()
        
        plt.bar(x + (i - len(metrics)/2 + 0.5) * width, values, width, label=label)
    
    plt.xlabel("Configuration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Metrics by Configuration")
    plt.xticks(x, categories, rotation=45)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"gemini_Q4_accuracy_metrics_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # 2. Create a heatmap of accuracy metrics only for files with length constraints
    df_with_length = df[df["has_length_constraint"]].copy()
    # Only use numeric word length
    df_clean = df_with_length[pd.to_numeric(df_with_length['word_length'], errors='coerce').notna()].copy()
    
    if not df_clean.empty and len(df_clean) > 1:  # Need at least 2 rows for a meaningful heatmap
        plt.figure(figsize=(12, 8))
        
        # Create pivot tables for each metric
        metrics_plots = {}
        all_metrics = ["english_words_accuracy", "vowel_constraint_accuracy", 
                      "length_constraint_accuracy", "fully_valid_accuracy"]
        all_labels = ["English Words", "Valid Vowels", "Valid Length", "Fully Valid"]
        
        for metric, label in zip(all_metrics, all_labels):
            pivot = df_clean.pivot_table(
                index="word_count", 
                columns="word_length", 
                values=metric, 
                aggfunc='mean'
            )
            if not pivot.empty:
                metrics_plots[label] = pivot
        
        # Plot 2x2 heatmaps if we have enough metrics
        if len(metrics_plots) > 0:
            rows = (len(metrics_plots) + 1) // 2
            cols = min(2, len(metrics_plots))
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1 or cols == 1:
                axes = axes.reshape(rows, cols)
            
            i = 0
            for label, pivot in metrics_plots.items():
                row, col = i // cols, i % cols
                sns.heatmap(pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=1, 
                          ax=axes[row, col], fmt=".2f")
                axes[row, col].set_title(f"{label} Accuracy")
                axes[row, col].set_xlabel("Word Length")
                axes[row, col].set_ylabel("Word Count")
                i += 1
            
            # Hide empty subplots
            for j in range(i, rows * cols):
                row, col = j // cols, j % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            output_file = f"gemini_Q4_accuracy_heatmap_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")

def plot_performance_metrics(df, timestamp):
    """Create visualizations for performance metrics."""
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Create config labels if they don't exist
    if "config_label" not in df.columns:
        df["config_label"] = df.apply(
            lambda x: f"{x['word_count']}W-{x['word_length']}L" if x["has_length_constraint"] 
            else f"{x['word_count']}W", axis=1
        )
    
    # Sort by word count and word length for better visualization
    df = df.sort_values(by=["word_count", "word_length"])
    
    # 1. Create a figure for performance metrics
    plt.figure(figsize=(15, 8))
    
    categories = df["config_label"].tolist()
    
    # Plot performance metrics
    plt.subplot(1, 2, 1)
    plt.bar(categories, df["total_duration_seconds"], color="skyblue")
    plt.xlabel("Configuration")
    plt.ylabel("Duration (seconds)")
    plt.title("Total Duration by Configuration")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    plt.bar(categories, df["uniqueness_ratio"], color="lightgreen")
    plt.xlabel("Configuration")
    plt.ylabel("Ratio")
    plt.title("Word Uniqueness Ratio by Configuration")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = f"gemini_Q4_performance_metrics_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # 2. Complexity vs. Performance (only for configurations with length constraints)
    df_with_length = df[df["has_length_constraint"]].copy()
    df_clean = df_with_length.copy()
    
    # Convert word_length to numeric and drop rows with non-numeric values
    df_clean["word_length"] = pd.to_numeric(df_clean["word_length"], errors="coerce")
    df_clean = df_clean.dropna(subset=["word_length"])
    
    if not df_clean.empty and len(df_clean) > 1:  # Need at least 2 points for a trend line
        plt.figure(figsize=(12, 6))
        
        # Calculate complexity as word_count * word_length
        df_clean["complexity"] = df_clean["word_count"] * df_clean["word_length"]
        
        plt.subplot(1, 2, 1)
        plt.scatter(df_clean["complexity"], df_clean["total_duration_seconds"], 
                   s=80, alpha=0.7)
        
        # Add regression line
        if len(df_clean) > 2:  # Need at least 3 points for a meaningful trend
            # Make sure we're working with numeric data
            x = df_clean["complexity"].astype(float).values
            y = df_clean["total_duration_seconds"].astype(float).values
            
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                sorted_x = sorted(x)
                plt.plot(sorted_x, p(sorted_x), "r--", linewidth=2)
            except Exception as e:
                print(f"Warning: Could not create regression line for duration: {e}")
        
        plt.xlabel("Complexity (Word Count × Word Length)")
        plt.ylabel("Duration (seconds)")
        plt.title("Task Complexity vs. Duration")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(df_clean["complexity"], df_clean["uniqueness_ratio"], 
                   s=80, alpha=0.7)
        
        # Add regression line
        if len(df_clean) > 2:
            # Make sure we're working with numeric data
            x = df_clean["complexity"].astype(float).values
            y = df_clean["uniqueness_ratio"].astype(float).values
            
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                sorted_x = sorted(x)
                plt.plot(sorted_x, p(sorted_x), "r--", linewidth=2)
            except Exception as e:
                print(f"Warning: Could not create regression line for uniqueness: {e}")
        
        plt.xlabel("Complexity (Word Count × Word Length)")
        plt.ylabel("Uniqueness Ratio")
        plt.title("Task Complexity vs. Uniqueness")
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = f"gemini_Q4_complexity_analysis_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")

def analyze_common_words(word_data, timestamp):
    """Analyze common words across different configurations."""
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Create a figure to show the most common words for each configuration
    num_configs = len(word_data)
    
    if num_configs == 0:
        print("No word data to analyze.")
        return
    
    # Calculate rows and columns for subplots
    cols = min(2, num_configs)
    rows = (num_configs + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (config, words) in enumerate(sorted(word_data.items())):
        ax = plt.subplot(rows, cols, i + 1)
        
        if not words:
            ax.text(0.5, 0.5, "No common words data", 
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            continue
        
        # Extract words and frequencies
        words_list = [item[0] for item in words]
        freqs = [item[1] for item in words]
        
        # Limit to top 10 words if there are many
        if len(words_list) > 10:
            words_list = words_list[:10]
            freqs = freqs[:10]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(words_list))
        ax.barh(y_pos, freqs, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words_list)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title(f"Most Common Words for {config}")
        ax.set_xlabel("Frequency")
    
    plt.tight_layout()
    output_file = f"gemini_Q4_common_words_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

def generate_summary_report(df, timestamp):
    """Generate a text summary report of the analysis."""
    # Calculate overall metrics
    overall_english_words = df["english_words_accuracy"].mean()
    overall_vowel_constraint = df["vowel_constraint_accuracy"].mean()
    overall_fully_valid = df["fully_valid_accuracy"].mean()
    overall_count_accuracy = df["count_accuracy"].mean()
    overall_success_rate = df["success_rate"].mean()
    
    # Format report
    report = [
        "# Gemini Q4 Vowel Constraint Test Analysis",
        f"## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall Performance",
        f"- Success Rate: {overall_success_rate:.2f}",
        f"- English Words Accuracy: {overall_english_words:.2f}",
        f"- Vowel Constraint Accuracy: {overall_vowel_constraint:.2f}"
    ]
    
    # Add length constraint metrics only if applicable
    if df["has_length_constraint"].any():
        # Calculate only for files with length constraints
        df_with_length = df[df["has_length_constraint"]]
        if not df_with_length.empty:
            length_constraint_mean = df_with_length["length_constraint_accuracy"].dropna().mean()
            report.append(f"- Length Constraint Accuracy: {length_constraint_mean:.2f}")
    
    report.extend([
        f"- Fully Valid Words Accuracy: {overall_fully_valid:.2f}",
        f"- Word Count Accuracy: {overall_count_accuracy:.2f}",
        "",
        "## Performance by Configuration"
    ])
    
    # Add details for each configuration
    for _, row in df.sort_values(by=["word_count", "word_length"]).iterrows():
        config = f"{row['word_count']}W"
        if row["has_length_constraint"]:
            config += f"-{row['word_length']}L"
            
        report.append(f"### {config}")
        report.append(f"- Success Rate: {row['success_rate']:.2f}")
        report.append(f"- English Words Accuracy: {row['english_words_accuracy']:.2f}")
        report.append(f"- Vowel Constraint Accuracy: {row['vowel_constraint_accuracy']:.2f}")
        
        if row["has_length_constraint"] and pd.notna(row["length_constraint_accuracy"]):
            report.append(f"- Length Constraint Accuracy: {row['length_constraint_accuracy']:.2f}")
            
        report.append(f"- Fully Valid Words Accuracy: {row['fully_valid_accuracy']:.2f}")
        report.append(f"- Uniqueness Ratio: {row['uniqueness_ratio']:.2f}")
        report.append(f"- Total Duration: {row['total_duration_seconds']:.2f} seconds")
        report.append("")
    
    # Write to file
    output_file = f"gemini_Q4_summary_report_{timestamp}.md"
    with open(output_file, "w") as f:
        f.write("\n".join(report))
    
    print(f"Saved: {output_file}")

def main():
    """Main function to execute the script."""
    print("Searching for Gemini Q4 test result files...")
    df, word_data = collect_metrics()
    
    if len(df) == 0:
        print("No matching files found. Please ensure files follow the pattern 'gemini_Q4_<word_count>W.json' or 'gemini_Q4_<word_count>W-<word_length>L.json'")
        return
    
    print(f"Found {len(df)} test result files.")
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Generating accuracy metrics visualizations...")
    plot_accuracy_metrics(df, timestamp)
    
    print("Generating performance metrics visualizations...")
    plot_performance_metrics(df, timestamp)
    
    print("Analyzing common words...")
    analyze_common_words(word_data, timestamp)
    
    print("Generating summary report...")
    generate_summary_report(df, timestamp)
    
    print("All visualizations and reports created successfully!")

if __name__ == "__main__":
    main()