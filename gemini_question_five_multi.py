import json
import os
from dotenv import load_dotenv
import re
import requests
from typing import List, Dict, Any
import statistics
import time

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
    starting_letter: str = "P", 
    word_length: int = 11, 
    word_count: int = 10
) -> Dict[str, Any]:
    # Create the prompt asking for words with specific criteria
    prompt = f"""
    Task: Give me a list of exactly {word_count} words that:
    1. Start with the letter '{starting_letter.upper()}' (or '{starting_letter.lower()}')
    2. Have exactly {word_length} letters each
    
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
        validation_results = validate_word_list(words, starting_letter, word_length, word_count)
        
        return {
            "success": True,
            "requested": {
                "starting_letter": starting_letter,
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
                "correct_words": 0,
                "criteria_accuracy": 0.0,
                "count_accuracy": 0.0,
                "overall_accuracy": 0.0,
                "errors": [f"Failed to parse response: {str(e)}"]
            }
        }

# Function to validate if words meet criteria
def validate_word_list(words: List[str], 
                        starting_letter: str, 
                        required_length: int, 
                        required_count: int) -> Dict[str, Any]:
    if not words:
        return {
            "total_words": 0,
            "correct_words": 0,
            "correct_count": False,
            "criteria_accuracy": 0.0,
            "count_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "errors": ["No words returned"]
        }
    
    # Convert to lowercase for case-insensitive comparison
    starting_letter = starting_letter.lower()
    
    # Validate each word against the criteria
    correct_words = []
    wrong_start_words = []
    wrong_length_words = []
    non_alpha_words = []
    
    for word in words:
        word = word.strip()
        
        if not word.isalpha():
            non_alpha_words.append(word)
            continue
            
        # Check if word starts with the correct letter
        correct_start = word.lower().startswith(starting_letter)
        
        # Check if word has the correct length
        correct_length = len(word) == required_length
        
        if correct_start and correct_length:
            correct_words.append(word)
        elif not correct_start:
            wrong_start_words.append(f"{word} (starts with '{word[0]}')")
        elif not correct_length:
            wrong_length_words.append(f"{word} ({len(word)} letters)")
    
    # Calculate accuracy metrics
    total_words = len(words)
    correct_word_count = len(correct_words)
    criteria_accuracy = correct_word_count / total_words if total_words > 0 else 0
    count_accuracy = 1.0 if total_words == required_count else total_words / required_count
    
    errors = []
    if wrong_start_words:
        errors.append(f"Words not starting with '{starting_letter.upper()}': {', '.join(wrong_start_words)}")
    if wrong_length_words:
        errors.append(f"Words with incorrect length: {', '.join(wrong_length_words)}")
    if non_alpha_words:
        errors.append(f"Non-alphabetic words: {', '.join(non_alpha_words)}")
    if total_words != required_count:
        errors.append(f"Expected {required_count} words, got {total_words}")
    
    return {
        "total_words": total_words,
        "correct_words": correct_word_count,
        "correct_count": total_words == required_count,
        "criteria_accuracy": criteria_accuracy,
        "count_accuracy": count_accuracy,
        "overall_accuracy": (criteria_accuracy + count_accuracy) / 2 if total_words > 0 else 0,
        "errors": errors if errors else []
    }

# Run multiple tests and calculate average metrics
def run_multiple_tests(
    api_key: str,
    starting_letter: str = "P", 
    word_length: int = 11, 
    word_count: int = 10,
    num_tests: int = 10
) -> Dict[str, Any]:
    all_results = []
    start_time = time.time()
    
    print(f"Running {num_tests} tests...")
    for i in range(num_tests):
        print(f"Test {i+1}/{num_tests}...")
        result = get_and_validate_word_list(api_key, starting_letter, word_length, word_count)
        all_results.append(result)
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    # Calculate average metrics
    total_words_list = []
    correct_words_list = []
    criteria_accuracy_list = []
    count_accuracy_list = []
    overall_accuracy_list = []
    successful_tests = 0
    total_unique_words = set()
    
    # Track all words returned and their frequency
    word_frequency = {}
    
    for result in all_results:
        if result.get("success", False):
            successful_tests += 1
            validation = result.get("validation", {})
            total_words_list.append(validation.get("total_words", 0))
            correct_words_list.append(validation.get("correct_words", 0))
            criteria_accuracy_list.append(validation.get("criteria_accuracy", 0))
            count_accuracy_list.append(validation.get("count_accuracy", 0))
            overall_accuracy_list.append(validation.get("overall_accuracy", 0))
            
            # Track all words
            words = result.get("response", {}).get("words", [])
            for word in words:
                total_unique_words.add(word)
                if word in word_frequency:
                    word_frequency[word] += 1
                else:
                    word_frequency[word] = 1
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate average metrics
    avg_metrics = {
        "num_tests": num_tests,
        "successful_tests": successful_tests,
        "success_rate": successful_tests / num_tests if num_tests > 0 else 0,
        "avg_total_words": statistics.mean(total_words_list) if total_words_list else 0,
        "avg_correct_words": statistics.mean(correct_words_list) if correct_words_list else 0,
        "avg_criteria_accuracy": statistics.mean(criteria_accuracy_list) if criteria_accuracy_list else 0,
        "avg_count_accuracy": statistics.mean(count_accuracy_list) if count_accuracy_list else 0,
        "avg_overall_accuracy": statistics.mean(overall_accuracy_list) if overall_accuracy_list else 0,
        "total_unique_words": len(total_unique_words),
        "most_common_words": sorted([(w, c) for w, c in word_frequency.items()], key=lambda x: x[1], reverse=True)[:10],
        "total_duration_seconds": total_duration
    }
    
    # Add standard deviation if we have enough data
    if len(criteria_accuracy_list) > 1:
        avg_metrics["std_dev_criteria_accuracy"] = statistics.stdev(criteria_accuracy_list)
        avg_metrics["std_dev_count_accuracy"] = statistics.stdev(count_accuracy_list)
        avg_metrics["std_dev_overall_accuracy"] = statistics.stdev(overall_accuracy_list)
    
    return {
        "avg_metrics": avg_metrics,
        "all_results": all_results
    }

# Main function to demonstrate the code
def main():
    # Parameters for the word list request
    starting_letter = "S"
    word_length = 10
    word_count = 10
    num_tests = 10
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
    
    # If API key is not found in environment variables, prompt the user
    if not api_key:
        print("GOOGLE_AI_STUDIO_API_KEY not found in .env file")
        api_key = input("Enter your Google AI Studio API key: ")
    
    print(f"Connecting to Google AI Studio API (using Gemini 2.0 Flash model) for {num_tests} tests...")
    
    # Run multiple tests and get average metrics
    results = run_multiple_tests(api_key, starting_letter, word_length, word_count, num_tests)
    
    # Print average metrics
    avg_metrics = results["avg_metrics"]
    
    print("\n=== Average Metrics Across All Tests ===")
    print(f"Tests run: {avg_metrics['num_tests']}")
    print(f"Successful tests: {avg_metrics['successful_tests']} ({avg_metrics['success_rate']:.2%})")
    print(f"Average words returned: {avg_metrics['avg_total_words']:.2f}")
    print(f"Average words meeting criteria: {avg_metrics['avg_correct_words']:.2f}")
    print(f"Average criteria accuracy: {avg_metrics['avg_criteria_accuracy']:.2%}")
    print(f"Average count accuracy: {avg_metrics['avg_count_accuracy']:.2%}")
    print(f"Average overall accuracy: {avg_metrics['avg_overall_accuracy']:.2%}")
    
    if "std_dev_overall_accuracy" in avg_metrics:
        print(f"Standard deviation of overall accuracy: {avg_metrics['std_dev_overall_accuracy']:.2%}")
    
    print(f"\nTotal unique words across all tests: {avg_metrics['total_unique_words']}")
    
    print("\nMost common words across all tests:")
    for word, count in avg_metrics['most_common_words']:
        print(f"- {word}: {count} occurrences")
    
    # Save the results to a file
    filename = f"gemini_Q5_{starting_letter}-{word_count}W-{word_length}L.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {filename}")

if __name__ == "__main__":
    main()