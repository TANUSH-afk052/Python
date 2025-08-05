
# Rule-Based AI CLI Bot - Day 3 (AI/ML Learning Journey)

This is a command-line AI assistant bot built in Python. It is rule-based (not ML-based yet) and responds to specific user inputs using conditional logic.

## 🔧 Features Implemented (as of Day 3)

- Greets the user.
- Responds to basic commands like `search`, `open`, `github`, `wikipedia`, `help`.
- Uses `wikipedia` and `googlesearch-python` to get top results.
- Handles Wikipedia summaries.
- Opens URLs using `webbrowser`.
- Handles context-based follow-up:
  - `search` → prompts for search query.
  - `open github` → prompts for GitHub username.
- Responds without needing speech input.
- Uses `pyttsx3` for bot speech output.

## 📦 Libraries Used

- `wikipedia`
- `googlesearch-python`
- `webbrowser`
- `pyttsx3` (for text-to-speech)
- `os`, `sys`, `re` (for command handling and system ops)

## 🚀 Example Interaction

```
You: search
BOT: What would you like to search for on Google and Wikipedia?
You: Python programming
BOT: According to Wikipedia: Python is an interpreted, high-level and general-purpose programming language...
BOT: Here are the top 5 Google search results:
1. https://realpython.com/
2. https://www.python.org/
...
```

```
You: open github
BOT: Please enter the GitHub username.
You: thanos-code
BOT: Opening GitHub profile: https://github.com/thanos-code
```

## 🧠 Planned Improvements

- Add intent recognition using ML.
- Add chatbot memory using embeddings.
- Introduce spaCy/NLTK for NER or sentence understanding.
- Switch from rule-based to hybrid ML-NLP system.

---

