import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from datetime import datetime

def extract_w_pattern(filename):
    """Extract word count from Q2 filename pattern."""
    # Pattern to match 'gemini_Q2_5W.json'
    match = re.search(r'gemini_Q2_(\d+)W', filename)
    if match:
        return int(match.group(1))
    return None

def read_json_file(filepath):
    """Read a JSON file and return the data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def collect_metrics(file_pattern="gemini_Q2_*W.json"):
    """Collect metrics from all matching files."""
    results = []
    bible_word_counts = {}
    
    for filepath in glob(file_pattern):
        word_count = extract_w_pattern(filepath)
        if word_count is None:
            print(f"Could not extract pattern from: {filepath}")
            continue
            
        data = read_json_file(filepath)
        
        # Get the model name and test configuration
        model = data["test_configuration"]["model"]
        
        # Get the metrics
        metrics = data["average_metrics"]
        
        result = {
            "model": model,
            "word_count": word_count,
            "word_count_accuracy": metrics["word_count_accuracy"],
            "bible_words_accuracy": metrics["bible_words_accuracy"],
            "total_success_rate": metrics["total_success_rate"],
            "test_duration": data["test_configuration"]["total_duration_seconds"],
            "num_tests": metrics["num_tests"]
        }
        
        # Collect individual test results for detailed analysis
        bible_words_data = {}
        for test in data["individual_test_results"]:
            if "bible_words_found" in test["validation"]:
                for word in test["validation"]["bible_words_found"]:
                    if word in bible_words_data:
                        bible_words_data[word] += 1
                    else:
                        bible_words_data[word] = 1
        
        bible_word_counts[word_count] = bible_words_data
        results.append(result)
    
    return pd.DataFrame(results), bible_word_counts

def plot_accuracy_metrics(df):
    """Create visualizations for accuracy metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # 1. Accuracy Bar Chart
    plt.figure(figsize=(12, 8))
    
    # Sort by word count for better visualization
    df = df.sort_values(by="word_count")
    
    x = np.arange(len(df))
    width = 0.25
    
    plt.bar(x - width, df["word_count_accuracy"], width, label="Word Count Accuracy")
    plt.bar(x, df["bible_words_accuracy"], width, label="Bible Words Accuracy")
    plt.bar(x + width, df["total_success_rate"], width, label="Total Success Rate")
    
    plt.xlabel("Required Word Count")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Metrics by Required Word Count")
    plt.xticks(x, df["word_count"])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f"gemini_Q2_accuracy_metrics_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # 2. Line plot showing accuracy trends
    plt.figure(figsize=(10, 6))
    
    plt.plot(df["word_count"], df["word_count_accuracy"], 
             marker='o', linewidth=2, markersize=10, label="Word Count Accuracy")
    plt.plot(df["word_count"], df["bible_words_accuracy"], 
             marker='s', linewidth=2, markersize=10, label="Bible Words Accuracy")
    plt.plot(df["word_count"], df["total_success_rate"], 
             marker='^', linewidth=2, markersize=10, label="Total Success Rate")
    
    plt.xlabel("Required Word Count")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Trends by Required Word Count")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.legend()
    
    output_file = f"gemini_Q2_accuracy_trends_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

def plot_performance_metrics(df, bible_word_counts):
    """Create visualizations for performance metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # 1. Test Duration by Word Count
    plt.figure(figsize=(10, 6))
    
    # Sort by word count for better visualization
    df = df.sort_values(by="word_count")
    
    # Bar chart for test duration
    plt.bar(df["word_count"], df["test_duration"], color="skyblue", alpha=0.7)
    
    # Add a line showing duration per test
    plt.plot(df["word_count"], df["test_duration"] / df["num_tests"], 
             marker='o', linewidth=2, markersize=10, color="red", label="Duration per Test")
    
    plt.xlabel("Required Word Count")
    plt.ylabel("Duration (seconds)")
    plt.title("Test Duration by Required Word Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = f"gemini_Q2_duration_metrics_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # 2. Most Common Bible Words Found
    plt.figure(figsize=(12, 8))
    
    # For each word count, get top N bible words
    top_n = 10
    plots_needed = len(bible_word_counts)
    cols = min(2, plots_needed)
    rows = (plots_needed + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])  # Make it indexable
    
    subplot_idx = 0
    for word_count, word_data in sorted(bible_word_counts.items()):
        if not word_data:  # Skip if empty
            continue
            
        row = subplot_idx // cols
        col = subplot_idx % cols
        
        # Get current axis (handle both 1D and 2D cases)
        if rows == 1:
            ax = axes[col]
        elif cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        # Sort by frequency and get top N
        sorted_words = sorted(word_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words, counts = zip(*sorted_words) if sorted_words else ([], [])
        
        # Create horizontal bar chart
        y_pos = np.arange(len(words))
        ax.barh(y_pos, counts, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title(f"Top Bible Words ({word_count}-word sentences)")
        ax.set_xlabel("Frequency")
        
        subplot_idx += 1
    
    # Hide empty subplots
    for i in range(subplot_idx, rows * cols):
        row = i // cols
        col = i % cols
        if rows == 1:
            axes[col].axis('off')
        elif cols == 1:
            axes[row].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    output_file = f"gemini_Q2_bible_words_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

def analyze_failure_patterns(bible_word_counts):
    """Create a summary of the most frequent bible words found."""
    # Combine all word counts across different test configurations
    all_words = {}
    for word_data in bible_word_counts.values():
        for word, count in word_data.items():
            if word in all_words:
                all_words[word] += count
            else:
                all_words[word] = count
    
    # Sort by frequency
    sorted_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)
    
    # Create a bar chart of the top N most common words
    top_n = 20
    top_words = sorted_words[:top_n]
    
    plt.figure(figsize=(12, 8))
    words, counts = zip(*top_words) if top_words else ([], [])
    y_pos = np.arange(len(words))
    
    plt.barh(y_pos, counts, align='center')
    plt.yticks(y_pos, words)
    plt.xlabel('Frequency')
    plt.title(f'Top {top_n} Most Common Bible Words in Generated Sentences')
    plt.gca().invert_yaxis()  # Labels read top-to-bottom
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"gemini_Q2_common_bible_words_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Return the top words for the text report
    return top_words

def generate_summary_report(df, top_words):
    """Generate a text summary report of the analysis."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate overall performance
    overall_word_count_accuracy = df["word_count_accuracy"].mean()
    overall_bible_words_accuracy = df["bible_words_accuracy"].mean()
    overall_success_rate = df["total_success_rate"].mean()
    
    # Format report
    report = [
        "# Gemini Q2 Biblical Words Test Analysis",
        f"## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall Performance",
        f"- Word Count Accuracy: {overall_word_count_accuracy:.2f}",
        f"- Bible Words Accuracy: {overall_bible_words_accuracy:.2f}",
        f"- Total Success Rate: {overall_success_rate:.2f}",
        "",
        "## Performance by Configuration"
    ]
    
    # Add details for each configuration
    for _, row in df.sort_values(by="word_count").iterrows():
        report.append(f"### {row['word_count']}-word sentences")
        report.append(f"- Word Count Accuracy: {row['word_count_accuracy']:.2f}")
        report.append(f"- Bible Words Accuracy: {row['bible_words_accuracy']:.2f}")
        report.append(f"- Total Success Rate: {row['total_success_rate']:.2f}")
        report.append(f"- Test Duration: {row['test_duration']:.2f} seconds")
        report.append("")
    
    # Add information about the most common bible words
    report.append("## Most Common Bible Words Found")
    for i, (word, count) in enumerate(top_words[:10], 1):
        report.append(f"{i}. '{word}' (found {count} times)")
    
    # Write to file
    output_file = f"gemini_Q2_summary_report_{timestamp}.md"
    with open(output_file, "w") as f:
        f.write("\n".join(report))
    
    print(f"Saved: {output_file}")

def main():
    """Main function to execute the script."""
    print("Searching for Gemini Q2 test result files...")
    df, bible_word_counts = collect_metrics()
    
    if len(df) == 0:
        print("No matching files found. Please ensure files follow the pattern 'gemini_Q2_<word_count>W.json'.")
        return
    
    print(f"Found {len(df)} test result files.")
    
    print("Generating accuracy metrics visualizations...")
    plot_accuracy_metrics(df)
    
    print("Generating performance metrics visualizations...")
    plot_performance_metrics(df, bible_word_counts)
    
    print("Analyzing failure patterns...")
    top_words = analyze_failure_patterns(bible_word_counts)
    
    print("Generating summary report...")
    generate_summary_report(df, top_words)
    
    print("All visualizations and reports created successfully!")

if __name__ == "__main__":
    main()