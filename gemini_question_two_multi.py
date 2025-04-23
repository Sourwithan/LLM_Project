import json
import os
import re
import time
import datetime
import random
from typing import Dict, Any, List, Set
import requests

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

# Function to load Bible words from JSON file
def load_bible_words(file_path: str) -> Set[str]:
    """Load Bible words from a JSON file and return as a set for efficient lookup."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            bible_words_data = json.load(f)
            
        # Convert to lowercase for case-insensitive comparison
        if isinstance(bible_words_data, list):
            return {word.lower() for word in bible_words_data if isinstance(word, str)}
        elif isinstance(bible_words_data, dict):
            # If it's a dictionary, extract words based on the structure
            # Adjust this based on the actual structure of your JSON file
            all_words = set()
            for key, value in bible_words_data.items():
                if isinstance(value, list):
                    all_words.update(word.lower() for word in value if isinstance(word, str))
                elif isinstance(value, str):
                    all_words.add(value.lower())
            return all_words
        else:
            print(f"Unexpected format in bible_words.json: {type(bible_words_data)}")
            return set()
    except Exception as e:
        print(f"Error loading Bible words: {str(e)}")
        return set()

# Function to validate if a sentence meets requirements (word count and no Bible words)
def validate_sentence(
    sentence: str, 
    bible_words: Set[str], 
    required_word_count: int = 10
) -> Dict[str, Any]:
    # Clean and tokenize the sentence (remove punctuation and split)
    cleaned_sentence = re.sub(r'[^\w\s]', '', sentence).strip()
    words = [word for word in cleaned_sentence.split() if word]
    
    # Check word count
    word_count = len(words)
    correct_word_count = word_count == required_word_count
    
    # Check for Bible words
    bible_words_found = []
    for word in words:
        if word.lower() in bible_words:
            bible_words_found.append(word)
    
    # Overall success
    success = correct_word_count and len(bible_words_found) == 0
    
    return {
        "original_sentence": sentence.strip(),
        "cleaned_sentence": cleaned_sentence,
        "words": words,
        "word_count": word_count,
        "correct_word_count": correct_word_count,
        "bible_words_found": bible_words_found,
        "has_bible_words": len(bible_words_found) > 0,
        "success": success
    }

# Function to get and validate a sentence
def get_and_validate_sentence(
    api_key: str,
    bible_words: Set[str],
    required_word_count: int = 10
) -> Dict[str, Any]:
    # Create the prompt asking for a sentence meeting the criteria
    prompt = f"""
    Write me a {required_word_count} word sentence without any words that appear in the Christian Bible.
    Only respond with the sentence itself, no additional explanation.
    """
    
    # Get response from Google AI Studio API
    content = get_gemini_response(prompt, api_key)
    
    # Validate the sentence
    validation_results = validate_sentence(content, bible_words, required_word_count)
    
    return {
        "success": validation_results["success"],
        "requested": {
            "task": f"Generate a {required_word_count}-word sentence without biblical words",
            "required_word_count": required_word_count
        },
        "response": {
            "raw_response": content
        },
        "validation": validation_results
    }

# Run multiple tests and calculate average metrics
def run_multiple_tests(
    api_key: str,
    bible_words: Set[str],
    required_word_count: int = 10,
    num_tests: int = 10
) -> Dict[str, Any]:
    all_results = []
    start_time = time.time()
    
    print(f"Running {num_tests} tests...")
    for i in range(num_tests):
        test_start_time = time.time()
        print(f"Test {i+1}/{num_tests}...")
        result = get_and_validate_sentence(api_key, bible_words, required_word_count)
        test_end_time = time.time()
        
        # Add timing information to the result
        result["timing"] = {
            "test_number": i + 1,
            "start_time": datetime.datetime.now().isoformat(),
            "duration_seconds": test_end_time - test_start_time
        }
        
        all_results.append(result)
        
        # Add a small delay to avoid rate limiting
        if i < num_tests - 1:  # No need to wait after the last test
            delay = random.uniform(3, 5)
            print(f"Waiting {delay:.2f} seconds before next test...")
            time.sleep(delay)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate success metrics
    word_count_correct = 0
    no_bible_words = 0
    fully_successful = 0
    
    for result in all_results:
        validation = result.get("validation", {})
        if validation.get("correct_word_count", False):
            word_count_correct += 1
        if not validation.get("has_bible_words", True):
            no_bible_words += 1
        if validation.get("success", False):
            fully_successful += 1
    
    # Average metrics
    avg_metrics = {
        "num_tests": num_tests,
        "word_count_accuracy": word_count_correct / num_tests if num_tests > 0 else 0,
        "bible_words_accuracy": no_bible_words / num_tests if num_tests > 0 else 0,
        "total_success_rate": fully_successful / num_tests if num_tests > 0 else 0
    }
    
    # Create a comprehensive report
    comprehensive_report = {
        "test_configuration": {
            "model": "gemini-2.0-flash",
            "prompt": f"Write me a {required_word_count} word sentence without any words that appear in the Christian Bible.",
            "required_word_count": required_word_count,
            "num_tests": num_tests,
            "total_duration_seconds": total_duration,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "average_metrics": avg_metrics,
        "individual_test_results": all_results
    }
    
    return comprehensive_report

# Main function to demonstrate the code
def main():
    # Parameters for the sentence request
    required_word_count = 5
    num_tests = 10  # Reduced from 10 to avoid rate limits
    
    # Get API key from environment or user input
    api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        api_key = input("Enter your Google AI Studio API key: ")
    
    # Load Bible words
    bible_words_path = "bible_words.json"
    if not os.path.exists(bible_words_path):
        print(f"Warning: Bible words file not found at {bible_words_path}")
        bible_words = set()
    else:
        print(f"Loading Bible words from {bible_words_path}...")
        bible_words = load_bible_words(bible_words_path)
        print(f"Loaded {len(bible_words)} unique Bible words")
    
    print(f"Connecting to Google AI Studio API for {num_tests} tests...")
    print(f"Testing prompt: 'Write me a {required_word_count} word sentence without any words that appear in the Christian Bible'")
    
    # Run multiple tests and get comprehensive report
    comprehensive_report = run_multiple_tests(api_key, bible_words, required_word_count, num_tests)
    
    # Print average metrics
    avg_metrics = comprehensive_report["average_metrics"]
    
    print("\n=== Average Metrics Across All Tests ===")
    print(f"Tests run: {avg_metrics['num_tests']}")
    print(f"Word count accuracy: {avg_metrics['word_count_accuracy']:.2%}")
    print(f"No Bible words accuracy: {avg_metrics['bible_words_accuracy']:.2%}")
    print(f"Overall success rate: {avg_metrics['total_success_rate']:.2%}")
    
    # Print individual test results
    print("\n=== Individual Test Results ===")
    for i, result in enumerate(comprehensive_report["individual_test_results"]):
        validation = result["validation"]
        print(f"\nTest {i+1}:")
        print(f"Sentence: \"{validation['original_sentence']}\"")
        print(f"Word count: {validation['word_count']} (Target: {required_word_count})")
        if validation['has_bible_words']:
            print(f"Bible words found: {', '.join(validation['bible_words_found'])}")
        else:
            print("No Bible words found")
        print(f"Overall success: {'Yes' if validation['success'] else 'No'}")
    
    # Save the comprehensive report to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_Q2_Result.json"
    with open(filename, "w") as f:
        json.dump(comprehensive_report, f, indent=2)
    print(f"\nComprehensive report saved to {filename}")

if __name__ == "__main__":
    main()