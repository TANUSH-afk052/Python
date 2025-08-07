import sqlite3
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PythonCodeDatabase:
    """Enhanced database for Python Q&A with better functionality"""
    
    def __init__(self, db_path='db/python_code_qa.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_db()
        self.populate_initial_data()
    
    def init_db(self):
        """Initialize database with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL UNIQUE,
                    answer TEXT NOT NULL,
                    source TEXT DEFAULT 'generated',
                    category TEXT DEFAULT 'general',
                    difficulty TEXT DEFAULT 'medium',
                    tags TEXT DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_question ON qa_pairs(question)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON qa_pairs(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags ON qa_pairs(tags)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON qa_pairs(created_at)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    def populate_initial_data(self):
        """Populate database with initial Python Q&A data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if data already exists
            cursor.execute('SELECT COUNT(*) FROM qa_pairs')
            count = cursor.fetchone()[0]
            
            if count > 0:
                conn.close()
                logger.info(f"Database already contains {count} Q&A pairs")
                return
            
            # Initial Python Q&A pairs with enhanced metadata
            initial_qa_pairs = [
                ("Write a Python function to check if a number is even", 
                 "def is_even(n):\n    \"\"\"Check if a number is even.\"\"\"\n    return n % 2 == 0", 
                 "initial", "basic", "easy", "functions,math"),
                
                ("Python function to find the maximum number in a list", 
                 "def find_max(numbers):\n    \"\"\"Find maximum number in a list.\"\"\"\n    if not numbers:\n        raise ValueError(\"List cannot be empty\")\n    return max(numbers)", 
                 "initial", "basic", "easy", "functions,lists"),
                
                ("Python function to calculate factorial of a number", 
                 "def factorial(n):\n    \"\"\"Calculate factorial using recursion.\"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial not defined for negative numbers\")\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)", 
                 "initial", "algorithms", "medium", "functions,recursion,math"),
                
                ("Python function to count vowels in a string", 
                 "def count_vowels(text):\n    \"\"\"Count vowels in a string.\"\"\"\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in text if char in vowels)", 
                 "initial", "strings", "easy", "functions,strings"),
                
                ("Python function to convert Celsius to Fahrenheit", 
                 "def celsius_to_fahrenheit(celsius):\n    \"\"\"Convert Celsius to Fahrenheit.\"\"\"\n    return (celsius * 9/5) + 32", 
                 "initial", "math", "easy", "functions,math,conversion"),
                
                ("Python function to check if a string contains only digits", 
                 "def is_numeric(text):\n    \"\"\"Check if string contains only digits.\"\"\"\n    return text.isdigit() if text else False", 
                 "initial", "strings", "easy", "functions,strings,validation"),
                
                ("Python function to remove duplicates from a list", 
                 "def remove_duplicates(lst):\n    \"\"\"Remove duplicates while preserving order.\"\"\"\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result", 
                 "initial", "lists", "medium", "functions,lists,sets"),
                
                ("Python function to generate Fibonacci sequence up to n terms", 
                 "def fibonacci(n):\n    \"\"\"Generate Fibonacci sequence.\"\"\"\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    \n    sequence = [0, 1]\n    for i in range(2, n):\n        sequence.append(sequence[i-1] + sequence[i-2])\n    return sequence", 
                 "initial", "algorithms", "medium", "functions,sequences,math"),
                
                ("Python function to check if a number is prime", 
                 "def is_prime(n):\n    \"\"\"Check if number is prime using optimized algorithm.\"\"\"\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    \n    for i in range(3, int(n ** 0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True", 
                 "initial", "algorithms", "medium", "functions,math,optimization"),
                
                ("Python function to solve Two Sum problem", 
                 "def two_sum(nums, target):\n    \"\"\"Solve Two Sum problem efficiently.\"\"\"\n    num_map = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in num_map:\n            return [num_map[complement], i]\n        num_map[num] = i\n    return []", 
                 "initial", "algorithms", "medium", "functions,hashmaps,leetcode"),
                
                ("Python function to check if a string is a valid palindrome", 
                 "def is_valid_palindrome(s):\n    \"\"\"Check if string is valid palindrome (alphanumeric only).\"\"\"\n    cleaned = ''.join(char.lower() for char in s if char.isalnum())\n    return cleaned == cleaned[::-1]", 
                 "initial", "strings", "medium", "functions,strings,algorithms"),
                
                ("Python function to find maximum subarray sum", 
                 "def max_subarray(nums):\n    \"\"\"Find maximum sum of contiguous subarray (Kadane's algorithm).\"\"\"\n    if not nums:\n        return 0\n    \n    max_sum = current_sum = nums[0]\n    for num in nums[1:]:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum", 
                 "initial", "algorithms", "hard", "functions,dynamic-programming,kadane"),
                
                ("Python function to implement binary search", 
                 "def binary_search(arr, target):\n    \"\"\"Implement binary search algorithm.\"\"\"\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1", 
                 "initial", "algorithms", "medium", "functions,search,binary-search"),
                
                ("Python class to implement a stack", 
                 "class Stack:\n    \"\"\"Stack implementation using list.\"\"\"\n    \n    def __init__(self):\n        self.items = []\n    \n    def push(self, item):\n        self.items.append(item)\n    \n    def pop(self):\n        if self.is_empty():\n            raise IndexError(\"Stack is empty\")\n        return self.items.pop()\n    \n    def peek(self):\n        if self.is_empty():\n            raise IndexError(\"Stack is empty\")\n        return self.items[-1]\n    \n    def is_empty(self):\n        return len(self.items) == 0\n    \n    def size(self):\n        return len(self.items)", 
                 "initial", "data-structures", "medium", "classes,stack,data-structures"),
                
                ("Python function to read file content", 
                 "def read_file(filename):\n    \"\"\"Read file content safely.\"\"\"\n    try:\n        with open(filename, 'r', encoding='utf-8') as f:\n            return f.read()\n    except FileNotFoundError:\n        raise FileNotFoundError(f\"File {filename} not found\")\n    except IOError as e:\n        raise IOError(f\"Error reading file: {e}\")", 
                 "initial", "file-io", "easy", "functions,files,io,error-handling")
            ]
            
            # Insert initial data
            cursor.executemany('''
                INSERT INTO qa_pairs 
                (question, answer, source, category, difficulty, tags) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', initial_qa_pairs)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database populated with {len(initial_qa_pairs)} initial Q&A pairs!")
            
        except Exception as e:
            logger.error(f"Failed to populate initial data: {str(e)}")
            raise
    
    def save_qa(self, question, answer, source='generated', category='general', difficulty='medium', tags=''):
        """Save Q&A pair to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Try to update existing question first
            cursor.execute('''
                UPDATE qa_pairs 
                SET answer = ?, source = ?, category = ?, difficulty = ?, tags = ?, updated_at = ?
                WHERE question = ?
            ''', (answer, source, category, difficulty, tags, datetime.now(), question))
            
            # If no rows were updated, insert new record
            if cursor.rowcount == 0:
                cursor.execute('''
                    INSERT INTO qa_pairs 
                    (question, answer, source, category, difficulty, tags) 
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (question, answer, source, category, difficulty, tags))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Q&A pair saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save Q&A pair: {str(e)}")
            raise
    
    def get_answer(self, question):
        """Get answer for a question"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT answer, id FROM qa_pairs 
                WHERE question = ? 
                ORDER BY updated_at DESC 
                LIMIT 1
            ''', (question,))
            
            result = cursor.fetchone()
            
            if result:
                answer, qa_id = result
                # Update usage count
                cursor.execute('''
                    UPDATE qa_pairs 
                    SET usage_count = usage_count + 1 
                    WHERE id = ?
                ''', (qa_id,))
                conn.commit()
            
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Failed to get answer: {str(e)}")
            return None
    
    def search_questions(self, keyword, limit=10):
        """Search questions by keyword"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT question, answer, category, difficulty, tags 
                FROM qa_pairs 
                WHERE question LIKE ? OR tags LIKE ? 
                ORDER BY usage_count DESC, updated_at DESC 
                LIMIT ?
            ''', (f'%{keyword}%', f'%{keyword}%', limit))
            
            results = cursor.fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search questions: {str(e)}")
            return []
    
    def get_recent_qa(self, limit=10):
        """Get recent Q&A pairs"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT question, answer, created_at 
                FROM qa_pairs 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get recent Q&A: {str(e)}")
            return []
    
    def get_stats(self):
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total pairs
            cursor.execute('SELECT COUNT(*) FROM qa_pairs')
            total_pairs = cursor.fetchone()[0]
            
            # Generated vs database answers
            cursor.execute('SELECT COUNT(*) FROM qa_pairs WHERE source = "generated"')
            generated_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM qa_pairs WHERE source != "generated"')
            database_count = cursor.fetchone()[0]
            
            # Most popular questions
            cursor.execute('''
                SELECT question, usage_count 
                FROM qa_pairs 
                WHERE usage_count > 0 
                ORDER BY usage_count DESC 
                LIMIT 5
            ''')
            popular_questions = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_pairs': total_pairs,
                'generated_count': generated_count,
                'database_count': database_count,
                'popular_questions': popular_questions
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {
                'total_pairs': 0,
                'generated_count': 0,
                'database_count': 0,
                'popular_questions': []
            }
    
    def get_by_category(self, category, limit=10):
        """Get Q&A pairs by category"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT question, answer, difficulty, tags 
                FROM qa_pairs 
                WHERE category = ? 
                ORDER BY updated_at DESC 
                LIMIT ?
            ''', (category, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get by category: {str(e)}")
            return []
    
    def get_categories(self):
        """Get all available categories"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT DISTINCT category, COUNT(*) 
                FROM qa_pairs 
                GROUP BY category 
                ORDER BY COUNT(*) DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get categories: {str(e)}")
            return []