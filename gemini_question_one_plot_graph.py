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
    """Extract word count and letter count from filename."""
    # Pattern to match 'gemini_Q1_5W_10L.json'
    match = re.search(r'gemini_Q1_(\d+)W_(\d+)L', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def read_json_file(filepath):
    """Read a JSON file and return the data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def collect_metrics(file_pattern="gemini_Q1_*W_*L.json"):
    """Collect metrics from all matching files."""
    results = []
    
    for filepath in glob(file_pattern):
        word_count, letter_count = extract_wl_pattern(filepath)
        if word_count is None or letter_count is None:
            print(f"Could not extract pattern from: {filepath}")
            continue
            
        data = read_json_file(filepath)
        
        # Get the model name from the data
        model = data["test_configuration"]["model"]
        
        # Get the metrics
        metrics = data["average_metrics"]
        
        result = {
            "model": model,
            "word_count": word_count,
            "letter_count": letter_count,
            "success_rate": metrics["success_rate"],
            "length_accuracy": metrics["avg_length_accuracy"],
            "count_accuracy": metrics["avg_count_accuracy"],
            "overall_accuracy": metrics["avg_overall_accuracy"],
            "total_unique_words": metrics["total_unique_words"],
            "test_duration": data["test_configuration"]["total_duration_seconds"]
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def plot_accuracy_metrics(df):
    """Create visualizations for accuracy metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # 1. Accuracy Heatmap
    plt.figure(figsize=(12, 10))
    
    plt.subplot(221)
    pivot_length = df.pivot_table(index="word_count", columns="letter_count", 
                                 values="length_accuracy", aggfunc='mean')
    sns.heatmap(pivot_length, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
    plt.title("Length Accuracy")
    plt.xlabel("Letter Count")
    plt.ylabel("Word Count")
    
    plt.subplot(222)
    pivot_count = df.pivot_table(index="word_count", columns="letter_count", 
                                values="count_accuracy", aggfunc='mean')
    sns.heatmap(pivot_count, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
    plt.title("Count Accuracy")
    plt.xlabel("Letter Count")
    plt.ylabel("Word Count")
    
    plt.subplot(223)
    pivot_overall = df.pivot_table(index="word_count", columns="letter_count", 
                                  values="overall_accuracy", aggfunc='mean')
    sns.heatmap(pivot_overall, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
    plt.title("Overall Accuracy")
    plt.xlabel("Letter Count")
    plt.ylabel("Word Count")
    
    plt.subplot(224)
    pivot_success = df.pivot_table(index="word_count", columns="letter_count", 
                                  values="success_rate", aggfunc='mean')
    sns.heatmap(pivot_success, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
    plt.title("Success Rate")
    plt.xlabel("Letter Count")
    plt.ylabel("Word Count")
    
    plt.tight_layout()
    plt.suptitle("Accuracy Metrics by Word Count and Letter Count", fontsize=16, y=1.02)
    output_file = f"gemini_Q1_accuracy_heatmaps_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # 2. Accuracy Line Plots
    plt.figure(figsize=(16, 8))
    
    # By Word Count
    word_groups = df.groupby("word_count").agg({
        "length_accuracy": "mean",
        "count_accuracy": "mean",
        "overall_accuracy": "mean",
        "success_rate": "mean"
    }).reset_index()
    
    plt.subplot(1, 2, 1)
    plt.plot(word_groups["word_count"], word_groups["length_accuracy"], 
             marker='o', linewidth=2, markersize=8, label="Length Accuracy")
    plt.plot(word_groups["word_count"], word_groups["count_accuracy"], 
             marker='s', linewidth=2, markersize=8, label="Count Accuracy")
    plt.plot(word_groups["word_count"], word_groups["overall_accuracy"], 
             marker='^', linewidth=2, markersize=8, label="Overall Accuracy")
    plt.plot(word_groups["word_count"], word_groups["success_rate"], 
             marker='*', linewidth=2, markersize=10, label="Success Rate")
    plt.grid(True, alpha=0.3)
    plt.title("Accuracy by Word Count")
    plt.xlabel("Word Count")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.legend()
    
    # By Letter Count
    letter_groups = df.groupby("letter_count").agg({
        "length_accuracy": "mean",
        "count_accuracy": "mean",
        "overall_accuracy": "mean",
        "success_rate": "mean"
    }).reset_index()
    
    plt.subplot(1, 2, 2)
    plt.plot(letter_groups["letter_count"], letter_groups["length_accuracy"], 
             marker='o', linewidth=2, markersize=8, label="Length Accuracy")
    plt.plot(letter_groups["letter_count"], letter_groups["count_accuracy"], 
             marker='s', linewidth=2, markersize=8, label="Count Accuracy")
    plt.plot(letter_groups["letter_count"], letter_groups["overall_accuracy"], 
             marker='^', linewidth=2, markersize=8, label="Overall Accuracy")
    plt.plot(letter_groups["letter_count"], letter_groups["success_rate"], 
             marker='*', linewidth=2, markersize=10, label="Success Rate")
    plt.grid(True, alpha=0.3)
    plt.title("Accuracy by Letter Count")
    plt.xlabel("Letter Count")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)
    plt.legend()
    
    plt.tight_layout()
    plt.suptitle("Accuracy Trends by Task Parameters", fontsize=16, y=1.02)
    output_file = f"gemini_Q1_accuracy_trends_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

def plot_performance_metrics(df):
    """Create visualizations for performance metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # 1. Performance heatmap
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    pivot_duration = df.pivot_table(index="word_count", columns="letter_count", 
                                   values="test_duration", aggfunc='mean')
    sns.heatmap(pivot_duration, annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title("Test Duration (seconds)")
    plt.xlabel("Letter Count")
    plt.ylabel("Word Count")
    
    plt.subplot(122)
    pivot_unique = df.pivot_table(index="word_count", columns="letter_count", 
                                 values="total_unique_words", aggfunc='mean')
    sns.heatmap(pivot_unique, annot=True, cmap="PuBu")
    plt.title("Total Unique Words Generated")
    plt.xlabel("Letter Count")
    plt.ylabel("Word Count")
    
    plt.tight_layout()
    plt.suptitle("Performance Metrics by Word Count and Letter Count", fontsize=16, y=1.02)
    output_file = f"gemini_Q1_performance_heatmaps_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # 2. Complexity vs Performance
    plt.figure(figsize=(12, 6))
    
    # Create complexity measure
    df["complexity"] = df["word_count"] * df["letter_count"]
    
    # Scatter plot with regression line
    plt.subplot(121)
    sns.regplot(x="complexity", y="test_duration", data=df, 
                scatter_kws={"s": 80}, line_kws={"color": "red"})
    plt.title("Task Complexity vs. Test Duration")
    plt.xlabel("Complexity (Word Count × Letter Count)")
    plt.ylabel("Test Duration (seconds)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(122)
    sns.regplot(x="complexity", y="total_unique_words", data=df, 
                scatter_kws={"s": 80}, line_kws={"color": "red"})
    plt.title("Task Complexity vs. Unique Words")
    plt.xlabel("Complexity (Word Count × Letter Count)")
    plt.ylabel("Total Unique Words")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle("Performance vs. Task Complexity", fontsize=16, y=1.02)
    output_file = f"gemini_Q1_performance_complexity_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

def main():
    """Main function to execute the script."""
    print("Searching for Gemini test result files...")
    df = collect_metrics()
    
    if len(df) == 0:
        print("No matching files found. Please ensure files follow the pattern 'gemini_Q1_<word_count>W_<letter_count>L.json'.")
        return
    
    print(f"Found {len(df)} test result files.")
    print("Generating accuracy metrics visualizations...")
    plot_accuracy_metrics(df)
    
    print("Generating performance metrics visualizations...")
    plot_performance_metrics(df)
    
    print("All visualizations created successfully!")

if __name__ == "__main__":
    main()