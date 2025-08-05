import pyttsx3
import webbrowser
import wikipedia
from googlesearch import search

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    print(f"BOT: {text}")
    engine.say(text)
    engine.runAndWait()

def search_google():
    speak("What should I search on Google?")
    query = input("You: ")
    speak(f"Searching Google for {query}...")
    results = list(search(query, num_results=5))
    if results:
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
        speak("Would you like to open one? Say the number or say no.")
        choice = input("You: ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= 5:
            webbrowser.open(results[int(choice)-1])
        else:
            speak("Okay, not opening anything.")
    else:
        speak("No results found.")

def search_wikipedia():
    speak("What topic should I look up on Wikipedia?")
    query = input("You: ")
    speak(f"Searching Wikipedia for {query}...")
    try:
        summary = wikipedia.summary(query, sentences=2)
        speak("According to Wikipedia:")
        print(summary)
        speak(summary)
    except wikipedia.exceptions.DisambiguationError as e:
        speak("There are multiple meanings, please be more specific.")
    except wikipedia.exceptions.PageError:
        speak("I couldn't find a page for that topic.")

def open_github():
    speak("Please enter the GitHub username to open:")
    username = input("You: ").strip()
    url = f"https://github.com/{username}"
    speak(f"Opening GitHub profile of {username}")
    webbrowser.open(url)

def open_instagram():
    speak("Please enter the Instagram username to open:")
    username = input("You: ").strip()
    url = f"https://instagram.com/{username}"
    speak(f"Opening Instagram profile of {username}")
    webbrowser.open(url)

def show_help():
    print("""
You can ask me things like:
  - search google
  - search wikipedia
  - open github
  - open instagram
  - help
  - quit
""")
    speak("These are some commands you can try.")

# Start Bot
speak("Hello! I am your AI CLI bot. How can I help you today?")

while True:
    command = input("You: ").lower().strip()

    if command == "search google":
        search_google()
    elif command == "search wikipedia":
        search_wikipedia()
    elif command == "open github":
        open_github()
    elif command == "open instagram":
        open_instagram()
    elif command in ("quit", "exit"):
        speak("Goodbye!")
        break
    elif command == "help":
        show_help()
    else:
        speak("Sorry, I didn't understand that. Type 'help' to see commands.")
