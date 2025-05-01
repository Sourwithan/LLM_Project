import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os

def read_json_file(filepath):
    """Read a JSON file and return the data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_accuracy_metrics(data, timestamp):
    """Create visualizations for accuracy metrics."""
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Extract metrics
    metrics = {
        "Success Rate": data["avg_metrics"]["success_rate"],
        "Yes/No Accuracy": data["avg_metrics"]["yes_no_accuracy_rate"],
        "Word Count Accuracy": data["avg_metrics"]["word_count_accuracy_rate"],
        "Uniqueness Ratio": data["avg_metrics"]["uniqueness_ratio"],
        "Legibility Yes Ratio": data["avg_metrics"]["legibility_yes_ratio"]
    }
    
    # 1. Create a bar chart for accuracy metrics
    plt.figure(figsize=(12, 8))
    
    # Sort keys for consistent ordering
    sorted_keys = sorted(metrics.keys())
    values = [metrics[key] for key in sorted_keys]
    
    bars = plt.bar(sorted_keys, values, color='skyblue', alpha=0.7)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.title("Accuracy Metrics")
    plt.ylabel("Score (0-1)")
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save the figure
    output_file = f"gemini_Q3_accuracy_metrics_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # 2. Create a pie chart for legibility distribution
    plt.figure(figsize=(10, 8))
    
    legibility = data["avg_metrics"]["legibility_distribution"]
    labels = list(legibility.keys())
    sizes = list(legibility.values())
    
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=['#66b3ff', '#ff9999'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title("Legibility Distribution")
    
    # Save the figure
    output_file = f"gemini_Q3_legibility_distribution_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

def plot_response_analysis(data, timestamp):
    """Create visualizations for response analysis."""
    # Set the style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Extract word counts from responses
    word_counts = []
    is_legible = []
    for result in data["all_results"]:
        word_counts.append(result["response"]["word_count"])
        is_legible.append(1 if result["response"]["still_legible"] == "Yes" else 0)
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        "Word Count": word_counts,
        "Is Legible": is_legible
    })
    
    # 1. Word count distribution
    plt.figure(figsize=(12, 6))
    
    # Count occurrences of each word count
    word_count_freq = df["Word Count"].value_counts().sort_index()
    
    plt.bar(word_count_freq.index, word_count_freq.values, color='lightgreen')
    plt.axvline(x=10, color='red', linestyle='--', label='Target (10 words)')
    
    plt.title("Distribution of Word Counts in Responses")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.xticks(range(min(word_counts), max(word_counts) + 1))
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save the figure
    output_file = f"gemini_Q3_word_count_distribution_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # 2. Relationship between word count and legibility
    plt.figure(figsize=(10, 6))
    
    # Group by word count and calculate legibility rate
    legibility_by_wordcount = df.groupby("Word Count")["Is Legible"].mean().reset_index()
    
    plt.bar(legibility_by_wordcount["Word Count"], legibility_by_wordcount["Is Legible"], 
            color='lightblue')
    plt.title("Legibility Rate by Word Count")
    plt.xlabel("Word Count")
    plt.ylabel("Legibility Rate (0-1)")
    plt.ylim(0, 1.1)
    plt.xticks(range(min(word_counts), max(word_counts) + 1))
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save the figure
    output_file = f"gemini_Q3_legibility_by_wordcount_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

def analyze_common_patterns(data, timestamp):
    """Analyze common patterns in responses."""
    # Set the style
    sns.set(style="ticks", font_scale=1.2)
    
    # Extract common sentences
    common_sentences = data["avg_metrics"]["most_common_sentences"]
    
    # Create a horizontal bar chart of common sentences
    plt.figure(figsize=(14, 8))
    
    sentences = [item[0] for item in common_sentences]
    frequencies = [item[1] for item in common_sentences]
    
    # Sort by frequency
    sorted_indices = np.argsort(frequencies)
    sentences = [sentences[i] for i in sorted_indices]
    frequencies = [frequencies[i] for i in sorted_indices]
    
    # Truncate long sentences for display
    display_sentences = []
    for s in sentences:
        if len(s) > 50:
            display_sentences.append(s[:47] + "...")
        else:
            display_sentences.append(s)
    
    y_pos = np.arange(len(display_sentences))
    plt.barh(y_pos, frequencies, align='center')
    plt.yticks(y_pos, display_sentences)
    plt.xlabel('Frequency')
    plt.title('Most Common Generated Sentences')
    
    # Save the figure
    output_file = f"gemini_Q3_common_sentences_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

def generate_summary_report(data, timestamp):
    """Generate a text summary report of the analysis."""
    # Format report
    report = [
        "# Gemini Q3 Regional Linguistic Test Analysis",
        f"## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall Performance",
        f"- Success Rate: {data['avg_metrics']['success_rate']:.2f}",
        f"- Yes/No Response Accuracy: {data['avg_metrics']['yes_no_accuracy_rate']:.2f}",
        f"- Word Count Accuracy: {data['avg_metrics']['word_count_accuracy_rate']:.2f}",
        f"- Uniqueness Ratio: {data['avg_metrics']['uniqueness_ratio']:.2f}",
        f"- Legibility Yes Ratio: {data['avg_metrics']['legibility_yes_ratio']:.2f}",
        "",
        "## Response Details",
        f"- Total Tests: {data['avg_metrics']['num_tests']}",
        f"- Successful Tests: {data['avg_metrics']['successful_tests']}",
        f"- Valid Yes/No Responses: {data['avg_metrics']['valid_yes_no_responses']}",
        f"- Correct Word Count Sentences: {data['avg_metrics']['correct_word_count_sentences']}",
        f"- Total Unique Sentences: {data['avg_metrics']['unique_sentences']}",
        "",
        "## Legibility Distribution",
        f"- Yes (Still Legible in New England): {data['avg_metrics']['legibility_distribution']['Yes']}",
        f"- No (Not Legible in New England): {data['avg_metrics']['legibility_distribution']['No']}",
        "",
        "## Most Common Sentences",
    ]
    
    # Add most common sentences
    for i, (sentence, count) in enumerate(data["avg_metrics"]["most_common_sentences"], 1):
        report.append(f"{i}. \"{sentence}\" (frequency: {count})")
    
    # Add execution time
    report.append("")
    report.append(f"Total Duration: {data['avg_metrics']['total_duration_seconds']:.2f} seconds")
    
    # Write to file
    output_file = f"gemini_Q3_summary_report_{timestamp}.md"
    with open(output_file, "w") as f:
        f.write("\n".join(report))
    
    print(f"Saved: {output_file}")

def main():
    """Main function to execute the script."""
    input_file = "gemini_Q3_Result.json"
    
    print(f"Reading file: {input_file}")
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return
    
    try:
        data = read_json_file(input_file)
    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file.")
        return
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Generating accuracy metrics visualizations...")
    plot_accuracy_metrics(data, timestamp)
    
    print("Analyzing response patterns...")
    plot_response_analysis(data, timestamp)
    
    print("Analyzing common patterns...")
    analyze_common_patterns(data, timestamp)
    
    print("Generating summary report...")
    generate_summary_report(data, timestamp)
    
    print("All visualizations and reports created successfully!")

if __name__ == "__main__":
    main()