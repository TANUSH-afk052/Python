
# 🧠 CLI Rule-Based AI Bot (Text + Speech)

This is a **simple rule-based AI CLI bot** written in **Python**, which can perform various tasks like:
- Searching Google
- Fetching summaries from Wikipedia
- Opening GitHub and Instagram profiles
- Using text-to-speech (TTS) to talk to the user

---

## 📦 Features

✅ Text-to-Speech interaction using `pyttsx3`  
✅ Google search results using `googlesearch-python`  
✅ Wikipedia summaries using `wikipedia`  
✅ Opens web profiles (GitHub, Instagram) via browser  
✅ Clean, interactive command-line interface  
✅ Asks for input when needed (e.g., search term, username)

---

## ⚙️ Technologies Used

| Library               | Purpose                               |
|-----------------------|----------------------------------------|
| `pyttsx3`             | Text-to-Speech (offline support)       |
| `wikipedia`           | Wikipedia API for fetching summaries   |
| `webbrowser`          | Open URLs in default browser           |
| `googlesearch-python` | Fetch Google search results            |

---

## 📥 Installation

```bash
pip install pyttsx3 wikipedia googlesearch-python
```

---

## 🚀 How It Works

```python
# 1. Speak a welcome message
speak("Hello! I am your AI CLI bot. How can I help you today?")

# 2. Wait for user command
command = input("You: ").lower().strip()

# 3. Based on the command, perform the task
if command == "search google":
    search_google()

# 4. Each task (Google, Wikipedia, GitHub, Instagram) is handled in a function
# 5. Each action includes TTS + input + optional browser opening
```

---

## 💬 Commands You Can Use

- `search google` → Asks what to search and shows results
- `search wikipedia` → Fetches and reads out the summary
- `open github` → Opens GitHub profile after asking for username
- `open instagram` → Same as above for Instagram
- `help` → Shows available commands
- `quit` or `exit` → Ends the program

---

## 🧪 Sample Output

```
BOT: Hello! I am your AI CLI bot. How can I help you today?
You: search wikipedia
BOT: What topic should I look up on Wikipedia?
You: Python (programming language)
BOT: According to Wikipedia:
"Python is a high-level, general-purpose programming language..."
```

---

## 🔧 Improvements You Can Make

- Add speech-to-text with `speech_recognition`  
- Use OpenAI API for intelligent response generation  
- Add chatbot memory or state tracking  
- Add command logging or GUI version  
- Integrate social media APIs (GitHub, Instagram)
