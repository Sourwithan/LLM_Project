import json
import os
import re
import requests
import nltk
from typing import List, Dict, Any
import statistics
import time

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Get English words from NLTK corpus
nltk_words = set(word.lower() for word in nltk.corpus.words.words())

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

# Function to check if a word contains each vowel at most once
def has_vowels_at_most_once(word: str) -> bool:
    """Check if a word contains each vowel (a, e, i, o, u) at most once."""
    vowels = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0}
    
    for char in word.lower():
        if char in vowels:
            vowels[char] += 1
            if vowels[char] > 1:
                return False
    
    return True

# Function to check if word meets exact length constraint
def has_exact_length(word: str, word_length: int) -> bool:
    """Check if word has the exact specified length."""
    return len(word) == word_length

# Function to validate a word against criteria
def validate_word(word: str, word_length: int) -> Dict[str, Any]:
    """Validate a word against our criteria."""
    # Remove any punctuation or spaces
    clean_word = ''.join(c for c in word.lower() if c.isalpha())
    
    # Check if word exists in NLTK corpus
    is_valid_english = clean_word in nltk_words
    
    # Check vowel constraint
    has_valid_vowels = has_vowels_at_most_once(clean_word)
    
    # Check length constraint
    actual_length = len(clean_word)
    has_valid_length = actual_length == word_length
    
    # Count vowels in the word
    vowel_counts = {v: clean_word.count(v) for v in 'aeiou'}
    total_vowels = sum(vowel_counts.values())
    
    # Identify repeated vowels if any
    repeated_vowels = [v for v, count in vowel_counts.items() if count > 1]
    
    return {
        "word": clean_word,
        "is_valid_english": is_valid_english,
        "has_valid_vowels": has_valid_vowels, 
        "has_valid_length": has_valid_length,
        "actual_length": actual_length,
        "required_length": word_length,
        "total_vowels": total_vowels,
        "vowel_counts": vowel_counts,
        "repeated_vowels": repeated_vowels
    }

# Function to get vowel words list
def get_vowel_words_list(api_key: str, word_count: int = 10, word_length: int = 6) -> Dict[str, Any]:
    # Create the prompt asking for words with specific criteria
    prompt = f"""
    Make a list of {word_count} words that contain all the English vowels at most once with a word length of {word_length}.
    
    Format your response as valid JSON with a key "words" containing an array of the {word_count} words.
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
        
        # Validate each word
        word_validations = [validate_word(word, word_length) for word in words]
        
        # Count words meeting criteria
        valid_english_count = sum(1 for validation in word_validations if validation["is_valid_english"])
        valid_vowels_count = sum(1 for validation in word_validations if validation["has_valid_vowels"])
        valid_length_count = sum(1 for validation in word_validations if validation["has_valid_length"])
        fully_valid_count = sum(1 for validation in word_validations 
                               if validation["is_valid_english"] 
                               and validation["has_valid_vowels"]
                               and validation["has_valid_length"])
        
        return {
            "success": True,
            "requested": f"{word_count} words that contain all English vowels at most once with a word length of {word_length}",
            "response": {
                "words": words,
                "word_count": len(words),
                "raw_response": content
            },
            "validation": {
                "word_validations": word_validations,
                "valid_english_count": valid_english_count,
                "valid_vowels_count": valid_vowels_count,
                "valid_length_count": valid_length_count,
                "fully_valid_count": fully_valid_count,
                "correct_count": len(words) == word_count,
                "accuracy": {
                    "english_words": valid_english_count / len(words) if words else 0,
                    "vowel_constraint": valid_vowels_count / len(words) if words else 0,
                    "length_constraint": valid_length_count / len(words) if words else 0,
                    "fully_valid": fully_valid_count / len(words) if words else 0,
                    "count_accuracy": 1.0 if len(words) == word_count else len(words) / word_count
                }
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "raw_response": content
        }

# Run multiple tests and collect results
def run_multiple_tests(api_key: str, word_count: int = 10, word_length: int = 6, num_tests: int = 10) -> Dict[str, Any]:
    all_results = []
    start_time = time.time()
    
    print(f"Running {num_tests} tests for {word_count} words, each with a length of {word_length}...")
    for i in range(num_tests):
        print(f"Test {i+1}/{num_tests}...")
        result = get_vowel_words_list(api_key, word_count, word_length)
        all_results.append(result)
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    # Calculate metrics
    successful_tests = sum(1 for result in all_results if result.get("success", False))
    
    # Aggregate validation metrics
    total_words = 0
    valid_english_total = 0
    valid_vowels_total = 0
    valid_length_total = 0
    fully_valid_total = 0
    correct_count_total = 0
    
    # Track all words returned
    all_words = []
    
    for result in all_results:
        if result.get("success", False):
            words = result.get("response", {}).get("words", [])
            all_words.extend(words)
            
            validation = result.get("validation", {})
            word_count_returned = len(words)
            total_words += word_count_returned
            valid_english_total += validation.get("valid_english_count", 0)
            valid_vowels_total += validation.get("valid_vowels_count", 0)
            valid_length_total += validation.get("valid_length_count", 0)
            fully_valid_total += validation.get("fully_valid_count", 0)
            correct_count_total += 1 if validation.get("correct_count", False) else 0
    
    # Calculate word uniqueness
    unique_words = set(all_words)
    
    # Count word frequencies
    word_frequency = {}
    for word in all_words:
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate average metrics
    avg_metrics = {
        "num_tests": num_tests,
        "word_count_requested": word_count,
        "word_length": word_length,
        "successful_tests": successful_tests,
        "success_rate": successful_tests / num_tests if num_tests > 0 else 0,
        "avg_accuracy": {
            "english_words": valid_english_total / total_words if total_words > 0 else 0,
            "vowel_constraint": valid_vowels_total / total_words if total_words > 0 else 0,
            "length_constraint": valid_length_total / total_words if total_words > 0 else 0,
            "fully_valid": fully_valid_total / total_words if total_words > 0 else 0,
            "correct_count": correct_count_total / successful_tests if successful_tests > 0 else 0
        },
        "total_words": total_words,
        "unique_words": len(unique_words),
        "uniqueness_ratio": len(unique_words) / total_words if total_words > 0 else 0,
        "most_common_words": sorted([(w, c) for w, c in word_frequency.items()], key=lambda x: x[1], reverse=True)[:10],
        "total_duration_seconds": total_duration
    }
    
    return {
        "avg_metrics": avg_metrics,
        "all_results": all_results
    }

# Main function to run the tests
def main():
    # Parameters for the request
    word_count = 20  # Number of words to request
    word_length = 20  # Exact length for each word
    num_tests = 10   # Number of tests to run
    
    # Get API key from environment or user input
    api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        api_key = input("Enter your Google AI Studio API key: ")
    
    print(f"Connecting to Google AI Studio API (using Gemini 2.0 Flash model) for {num_tests} tests...")
    
    # Run multiple tests and get metrics
    results = run_multiple_tests(api_key, word_count, word_length, num_tests)
    
    # Print metrics
    avg_metrics = results["avg_metrics"]
    
    print("\n=== Metrics Across All Tests ===")
    print(f"Tests run: {avg_metrics['num_tests']}")
    print(f"Word count requested per test: {avg_metrics['word_count_requested']}")
    print(f"Word length requirement: exactly {avg_metrics['word_length']} letters")
    print(f"Successful tests: {avg_metrics['successful_tests']} ({avg_metrics['success_rate']:.2%})")
    print(f"Total words analyzed: {avg_metrics['total_words']}")
    print(f"Words that are valid English: {avg_metrics['avg_accuracy']['english_words']:.2%}")
    print(f"Words with vowels appearing at most once: {avg_metrics['avg_accuracy']['vowel_constraint']:.2%}")
    print(f"Words with correct length: {avg_metrics['avg_accuracy']['length_constraint']:.2%}")
    print(f"Words that meet all criteria: {avg_metrics['avg_accuracy']['fully_valid']:.2%}")
    print(f"Tests returning exactly {word_count} words: {avg_metrics['avg_accuracy']['correct_count']:.2%}")
    print(f"Unique words across all tests: {avg_metrics['unique_words']}")
    print(f"Uniqueness ratio: {avg_metrics['uniqueness_ratio']:.2%}")
    
    print("\nMost common words across all tests:")
    for word, count in avg_metrics['most_common_words']:
        validation = validate_word(word, word_length)
        status = []
        if validation["is_valid_english"]:
            status.append("Valid English")
        else:
            status.append("Not in NLTK dictionary")
        
        if validation["has_valid_vowels"]:
            status.append("Valid vowel usage")
        else:
            status.append(f"Has repeated vowels: {', '.join(validation['repeated_vowels'])}")
            
        if validation["has_valid_length"]:
            status.append(f"Valid length ({validation['actual_length']} letters)")
        else:
            status.append(f"Invalid length ({validation['actual_length']} letters, should be {word_length})")
        
        print(f"- \"{word}\": {count} occurrences ({', '.join(status)})")
        print(f"  Vowel counts: {', '.join([f'{v}: {c}' for v, c in validation['vowel_counts'].items() if c > 0])}")
    
    # Show detailed results for the first test
    if results["all_results"] and results["all_results"][0]["success"]:
        first_test = results["all_results"][0]
        print("\n=== Detailed Results for First Test ===")
        print(f"Words returned: {first_test['response']['word_count']} (Target: {word_count})")
        
        # Show validation results for each word
        print("\nWord validations:")
        for i, validation in enumerate(first_test["validation"]["word_validations"]):
            word = validation["word"]
            status = "✓" if validation["is_valid_english"] and validation["has_valid_vowels"] and validation["has_valid_length"] else "✗"
            issues = []
            
            if not validation["is_valid_english"]:
                issues.append("Not in NLTK dictionary")
            
            if not validation["has_valid_vowels"]:
                issues.append(f"Repeated vowels: {', '.join(validation['repeated_vowels'])}")
                
            if not validation["has_valid_length"]:
                issues.append(f"Wrong length: {validation['actual_length']} letters (should be {word_length})")
            
            print(f"{i+1}. {word} {status} - {', '.join(issues) if issues else 'Valid'}")
            print(f"   Vowels: {', '.join([f'{v}: {c}' for v, c in validation['vowel_counts'].items() if c > 0])}")
    
    # Save the results to a file
    filename = "gemini_Q4_Results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {filename}")

if __name__ == "__main__":
    main()