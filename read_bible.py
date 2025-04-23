import json
import re
import sys

def extract_words(filename):
    # Check if filename is provided
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} input_filename.txt")
        sys.exit(1)
    
    try:
        # Read the file
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Remove all numbers and split into words
        # This regex removes standalone numbers and numbers within words
        words = re.findall(r'[a-zA-Z]+', text)
        
        # Convert to lowercase and remove duplicates using a set
        unique_words = set(word.lower() for word in words)
        
        # Convert set to list for JSON serialization
        word_list = sorted(list(unique_words))
        
        # Create output JSON filename
        output_file = filename.rsplit('.', 1)[0] + '_words.json'
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump({"words": word_list}, json_file, indent=4)
        
        print(f"Successfully extracted {len(word_list)} unique words to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Use the first command-line argument as the filename
    if len(sys.argv) > 1:
        extract_words(sys.argv[1])
    else:
        print(f"Usage: python {sys.argv[0]} bible.txt")