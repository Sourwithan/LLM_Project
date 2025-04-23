import json
import os
import re
import requests
from typing import List, Dict, Any
import statistics
import time
from collections import Counter
import datetime

# Function to get a response from Google AI Studio API
def get_gemini_response(prompt: str, api_key: str) -> str:
    """Generate a response using Google AI Studio Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        # Extract the text from the response
        content = response.json()
        if "candidates" in content and len(content["candidates"]) > 0:
            if "content" in content["candidates"][0]:
                text = content["candidates"][0]["content"]["parts"][0]["text"]
                return text
            
        return "No valid response received"
    except Exception as e:
        print(f"Error making API request: {str(e)}")
        return f"API Error: {str(e)}"

# Function to get and validate word list
def get_and_validate_word_list(
    api_key: str,
    word_length: int, 
    word_count: int
) -> Dict[str, Any]:
    # Create the prompt asking for words with specific criteria
    prompt = f"""
    Give me a list of {word_count} words that have {word_length} letters.
    Format your response as valid JSON with a single key "words" containing an array of the words.
    Only return the JSON, with no additional explanation or commentary.
    """
    
    # Get response from Google AI Studio API
    content = get_gemini_response(prompt, api_key)
    
    # Try to parse the JSON response
    try:
        # Extract JSON from the response if it contains other text
        json_match = re.search(r'({.*})', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        data = json.loads(content)
        words = data.get("words", [])
        
        # Validate the words
        validation_results = validate_word_list(words, word_length, word_count)
        
        return {
            "success": True,
            "requested": {
                "word_length": word_length,
                "word_count": word_count
            },
            "response": {
                "words": words,
                "raw_response": content
            },
            "validation": validation_results
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "raw_response": content,
            "validation": {
                "total_words": 0,
                "correct_length_words": 0,
                "length_accuracy": 0.0,
                "count_accuracy": 0.0,
                "overall_accuracy": 0.0,
                "errors": [f"Failed to parse response: {str(e)}"]
            }
        }

# Function to validate if words meet criteria
def validate_word_list(words: List[str], 
                      required_length: int, 
                      required_count: int) -> Dict[str, Any]:
    if not words:
        return {
            "total_words": 0,
            "correct_length_words": 0,
            "correct_count": False,
            "length_accuracy": 0.0,
            "count_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "errors": ["No words returned"]
        }
    
    # Validate each word against the criteria
    correct_length_words = []
    wrong_length_words = []
    non_alpha_words = []
    
    for word in words:
        word = word.strip()
        
        if not word.isalpha():
            non_alpha_words.append(word)
            continue
            
        # Check if word has the correct length
        correct_length = len(word) == required_length
        
        if correct_length:
            correct_length_words.append(word)
        else:
            wrong_length_words.append(f"{word} ({len(word)} letters)")
    
    # Calculate accuracy metrics
    total_words = len(words)
    correct_words_count = len(correct_length_words)
    length_accuracy = correct_words_count / total_words if total_words > 0 else 0
    count_accuracy = 1.0 if total_words == required_count else total_words / required_count
    
    errors = []
    if wrong_length_words:
        errors.append(f"Words with incorrect length: {', '.join(wrong_length_words)}")
    if non_alpha_words:
        errors.append(f"Non-alphabetic words: {', '.join(non_alpha_words)}")
    if total_words != required_count:
        errors.append(f"Expected {required_count} words, got {total_words}")
    
    return {
        "total_words": total_words,
        "correct_length_words": correct_words_count,
        "correct_count": total_words == required_count,
        "length_accuracy": length_accuracy,
        "count_accuracy": count_accuracy,
        "overall_accuracy": (length_accuracy + count_accuracy) / 2 if total_words > 0 else 0,
        "errors": errors if errors else []
    }

# Run multiple tests and calculate average metrics
def run_multiple_tests(
    api_key: str,
    word_length: int, 
    word_count: int,
    num_tests: int
) -> Dict[str, Any]:
    all_results = []
    start_time = time.time()
    
    # Track all words returned and their frequency
    word_counter = Counter()
    correct_word_counter = Counter()
    
    print(f"Running {num_tests} tests...")
    for i in range(num_tests):
        test_start_time = time.time()
        print(f"Test {i+1}/{num_tests}...")
        result = get_and_validate_word_list(api_key, word_length, word_count)
        test_end_time = time.time()
        
        # Add timing information to the result
        result["timing"] = {
            "test_number": i + 1,
            "start_time": datetime.datetime.now().isoformat(),
            "duration_seconds": test_end_time - test_start_time
        }
        
        all_results.append(result)
        
        # Track words and their frequencies
        if result.get("success", False):
            words = result.get("response", {}).get("words", [])
            
            # Count all words
            word_counter.update(words)
            
            # Count only correct length words
            correct_words = [word for word in words if len(word.strip()) == word_length]
            correct_word_counter.update(correct_words)
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate average metrics
    total_words_list = []
    correct_words_list = []
    length_accuracy_list = []
    count_accuracy_list = []
    overall_accuracy_list = []
    successful_tests = 0
    
    for result in all_results:
        if result.get("success", False):
            successful_tests += 1
            validation = result.get("validation", {})
            total_words_list.append(validation.get("total_words", 0))
            correct_words_list.append(validation.get("correct_length_words", 0))
            length_accuracy_list.append(validation.get("length_accuracy", 0))
            count_accuracy_list.append(validation.get("count_accuracy", 0))
            overall_accuracy_list.append(validation.get("overall_accuracy", 0))
    
    # Calculate average metrics
    avg_metrics = {
        "num_tests": num_tests,
        "successful_tests": successful_tests,
        "success_rate": successful_tests / num_tests if num_tests > 0 else 0,
        "avg_total_words": statistics.mean(total_words_list) if total_words_list else 0,
        "avg_correct_words": statistics.mean(correct_words_list) if correct_words_list else 0,
        "avg_length_accuracy": statistics.mean(length_accuracy_list) if length_accuracy_list else 0,
        "avg_count_accuracy": statistics.mean(count_accuracy_list) if count_accuracy_list else 0,
        "avg_overall_accuracy": statistics.mean(overall_accuracy_list) if overall_accuracy_list else 0,
        "total_unique_words": len(word_counter),
        "total_unique_correct_words": len(correct_word_counter),
        "most_common_words": word_counter.most_common(20),
        "most_common_correct_words": correct_word_counter.most_common(20)
    }
    
    # Add standard deviation if we have enough data
    if len(length_accuracy_list) > 1:
        avg_metrics["std_dev_length_accuracy"] = statistics.stdev(length_accuracy_list)
        avg_metrics["std_dev_count_accuracy"] = statistics.stdev(count_accuracy_list)
        avg_metrics["std_dev_overall_accuracy"] = statistics.stdev(overall_accuracy_list)
    
    # Word frequency data
    word_frequency_data = {
        "all_words": dict(word_counter),
        "correct_length_words": dict(correct_word_counter),
        "word_count_by_length": {}
    }
    
    # Count words by length
    for word, count in word_counter.items():
        length = len(word)
        if length not in word_frequency_data["word_count_by_length"]:
            word_frequency_data["word_count_by_length"][length] = 0
        word_frequency_data["word_count_by_length"][length] += count
    
    # Create a comprehensive report
    comprehensive_report = {
        "test_configuration": {
            "model": "gemini-2.0-flash",
            "prompt": f"Give me a list of {word_count} words that have {word_length} letters.",
            "word_length_requested": word_length,
            "word_count_requested": word_count,
            "num_tests": num_tests,
            "total_duration_seconds": total_duration,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "average_metrics": avg_metrics,
        "word_frequencies": word_frequency_data,
        "individual_test_results": all_results
    }
    
    return comprehensive_report

# Main function to demonstrate the code
def main():
    # Parameters for the word list request
    word_length = 20
    word_count = 20
    num_tests = 10
    
    # Get API key from environment or user input
    api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        api_key = input("Enter your Google AI Studio API key: ")
    
    print(f"Connecting to Google AI Studio API (using Gemini 2.0 Flash model) for {num_tests} tests...")
    print(f"Testing prompt: 'Give me a list of {word_count} words that have {word_length} letters'")
    
    # Run multiple tests and get comprehensive report
    comprehensive_report = run_multiple_tests(api_key, word_length, word_count, num_tests)
    
    # Print average metrics
    avg_metrics = comprehensive_report["average_metrics"]
    
    print("\n=== Average Metrics Across All Tests ===")
    print(f"Tests run: {avg_metrics['num_tests']}")
    print(f"Successful tests: {avg_metrics['successful_tests']} ({avg_metrics['success_rate']:.2%})")
    print(f"Average words returned: {avg_metrics['avg_total_words']:.2f}")
    print(f"Average words with correct length: {avg_metrics['avg_correct_words']:.2f}")
    print(f"Average length accuracy: {avg_metrics['avg_length_accuracy']:.2%}")
    print(f"Average count accuracy: {avg_metrics['avg_count_accuracy']:.2%}")
    print(f"Average overall accuracy: {avg_metrics['avg_overall_accuracy']:.2%}")
    
    if "std_dev_overall_accuracy" in avg_metrics:
        print(f"Standard deviation of overall accuracy: {avg_metrics['std_dev_overall_accuracy']:.2%}")
    
    print(f"\nTotal unique words across all tests: {avg_metrics['total_unique_words']}")
    print(f"Total unique correct-length words: {avg_metrics['total_unique_correct_words']}")
    
    print("\nMost common words across all tests:")
    for word, count in avg_metrics['most_common_words'][:20]:
        print(f"- {word}: {count} occurrences")
    
    # Save the comprehensive report to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_Q1_Result.json"
    with open(filename, "w") as f:
        json.dump(comprehensive_report, f, indent=2)
    print(f"\nComprehensive report saved to {filename}")

if __name__ == "__main__":
    main()