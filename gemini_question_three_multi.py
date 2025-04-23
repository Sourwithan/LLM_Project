import json
import os
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

# Function to get regional sentence and legibility response
def get_regional_sentence(api_key: str) -> Dict[str, Any]:
    # Create the prompt asking for a sentence with specific criteria
    prompt = """
    Write me a 10 word sentence that is legible in Southern North America and illegible in New England North America? 
    Determine If the sentence is still legible in North America with Yes or No only.
    
    Format your response as valid JSON with these keys:
    - "sentence": The 10-word sentence you've created
    - "still_legible": Either "Yes" or "No"
    
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
        sentence = data.get("sentence", "")
        still_legible = data.get("still_legible", "")
        
        # Count words in the sentence
        word_count = len(sentence.split())
        is_ten_words = word_count == 10
        
        # Validate still_legible is "Yes" or "No"
        is_valid_response = still_legible in ["Yes", "No"]
        
        return {
            "success": True,
            "requested": "10 word sentence legible in Southern North America and illegible in New England",
            "response": {
                "sentence": sentence,
                "word_count": word_count,
                "is_ten_words": is_ten_words,
                "still_legible": still_legible,
                "is_valid_response": is_valid_response,
                "raw_response": content
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}",
            "raw_response": content
        }

# Run multiple tests and collect results
def run_multiple_tests(api_key: str, num_tests: int = 10) -> Dict[str, Any]:
    all_results = []
    start_time = time.time()
    
    print(f"Running {num_tests} tests...")
    for i in range(num_tests):
        print(f"Test {i+1}/{num_tests}...")
        result = get_regional_sentence(api_key)
        all_results.append(result)
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    # Calculate metrics
    successful_tests = sum(1 for result in all_results if result.get("success", False))
    
    # Track sentences and legibility responses
    sentences = []
    legibility_responses = []
    valid_responses = 0
    correct_word_count = 0
    
    for result in all_results:
        if result.get("success", False):
            response_data = result.get("response", {})
            sentence = response_data.get("sentence", "")
            is_ten_words = response_data.get("is_ten_words", False)
            still_legible = response_data.get("still_legible", "")
            is_valid_response = response_data.get("is_valid_response", False)
            
            if sentence:
                sentences.append(sentence)
            
            if still_legible:
                legibility_responses.append(still_legible)
                
            if is_valid_response:
                valid_responses += 1
                
            if is_ten_words:
                correct_word_count += 1
    
    # Calculate sentence uniqueness
    unique_sentences = set(sentences)
    
    # Count sentence frequencies
    sentence_frequency = {}
    for sentence in sentences:
        if sentence in sentence_frequency:
            sentence_frequency[sentence] += 1
        else:
            sentence_frequency[sentence] = 1
    
    # Count legibility response frequencies
    legibility_frequency = {}
    for response in legibility_responses:
        if response in legibility_frequency:
            legibility_frequency[response] += 1
        else:
            legibility_frequency[response] = 1
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate metrics
    avg_metrics = {
        "num_tests": num_tests,
        "successful_tests": successful_tests,
        "success_rate": successful_tests / num_tests if num_tests > 0 else 0,
        "valid_yes_no_responses": valid_responses,
        "yes_no_accuracy_rate": valid_responses / successful_tests if successful_tests > 0 else 0,
        "correct_word_count_sentences": correct_word_count,
        "word_count_accuracy_rate": correct_word_count / successful_tests if successful_tests > 0 else 0,
        "total_sentences": len(sentences),
        "unique_sentences": len(unique_sentences),
        "uniqueness_ratio": len(unique_sentences) / len(sentences) if sentences else 0,
        "most_common_sentences": sorted([(s, c) for s, c in sentence_frequency.items()], key=lambda x: x[1], reverse=True)[:5],
        "legibility_distribution": {k: v for k, v in legibility_frequency.items()},
        "legibility_yes_ratio": legibility_frequency.get("Yes", 0) / len(legibility_responses) if legibility_responses else 0,
        "total_duration_seconds": total_duration
    }
    
    return {
        "avg_metrics": avg_metrics,
        "all_results": all_results
    }

# Main function to run the tests
def main():
    # Parameters for the request
    num_tests = 10
    
    # Get API key from environment or user input
    api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        api_key = input("Enter your Google AI Studio API key: ")
    
    print(f"Connecting to Google AI Studio API (using Gemini 2.0 Flash model) for {num_tests} tests...")
    
    # Run multiple tests and get metrics
    results = run_multiple_tests(api_key, num_tests)
    
    # Print metrics
    avg_metrics = results["avg_metrics"]
    
    print("\n=== Metrics Across All Tests ===")
    print(f"Tests run: {avg_metrics['num_tests']}")
    print(f"Successful tests: {avg_metrics['successful_tests']} ({avg_metrics['success_rate']:.2%})")
    print(f"Valid Yes/No responses: {avg_metrics['valid_yes_no_responses']} ({avg_metrics['yes_no_accuracy_rate']:.2%})")
    print(f"Sentences with exactly 10 words: {avg_metrics['correct_word_count_sentences']} ({avg_metrics['word_count_accuracy_rate']:.2%})")
    print(f"Total sentences generated: {avg_metrics['total_sentences']}")
    print(f"Unique sentences: {avg_metrics['unique_sentences']}")
    print(f"Uniqueness ratio: {avg_metrics['uniqueness_ratio']:.2%}")
    
    print("\nLegibility response distribution:")
    for response, count in avg_metrics['legibility_distribution'].items():
        print(f"- {response}: {count} ({count/sum(avg_metrics['legibility_distribution'].values()):.2%})")
    
    print("\nMost common sentences:")
    for sentence, count in avg_metrics['most_common_sentences']:
        print(f"- \"{sentence}\" ({count} occurrences)")
        print(f"  Word count: {len(sentence.split())}")
    
    # Save the results to a file
    with open("gemini_Q3_Result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to gemini_Q3_Result.json")

if __name__ == "__main__":
    main()