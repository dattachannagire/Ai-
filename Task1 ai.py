# Import the re module for regular expressions
import re

# Define patterns and responses
# Create a dictionary where keys are regex patterns and values are responses
patterns_responses = {
    r'hi|hello|hey': 'Hello! How can I assist you today?',
    r'how are you': 'I am just a bot, but I am doing great! How about you?',
    r'what is your name': 'I am a chatbot created to assist you with your questions.',
    r'(.*) your (favorite|favourite) (.*)': 'I do not have preferences, but I enjoy helping you!',
    r'thank you|thanks': 'You are welcome! If you have more questions, feel free to ask.',
    r'bye|goodbye': 'Goodbye! Have a great day!'
}

# Default response for unmatched patterns
# Set a default response for when no pattern matches the user input
default_response = "I'm sorry, I don't understand that. Can you please rephrase?"

# Function to match user input to a pattern and return the corresponding response
def get_response(user_input):
    # Iterate over the dictionary of patterns and responses
    for pattern, response in patterns_responses.items():
        # Check if the user input matches the current pattern (case-insensitive)
        if re.search(pattern, user_input, re.IGNORECASE):
            # If a match is found, return the corresponding response
            return response
    # If no match is found, return the default response
    return default_response

# Main function to run the chatbot
def chatbot():
    # Print an initial greeting message
    print("Chatbot: Hi! I am your rule-based chatbot. Type 'bye' to exit.")
    while True:
        # Read user input
        user_input = input("You: ")
        # Check if the user wants to exit the chat
        if re.search(r'bye|goodbye', user_input, re.IGNORECASE):
            # Print a farewell message and break the loop to exit
            print("Chatbot: Goodbye! Have a great day!")
            break
        # Get the response based on user input
        response = get_response(user_input)
        # Print the chatbot's response
        print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == '__main__':
    # Call the chatbot function to start the interaction
    chatbot()
