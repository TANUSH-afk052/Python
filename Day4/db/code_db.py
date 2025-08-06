# /db/code_db.py
import sqlite3
import os

class CodeDatabase:
    def __init__(self, db_path='code_qa.db'):
        self.db_path = db_path
        self.init_db()
        self.populate_initial_data()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def populate_initial_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM qa_pairs')
        count = cursor.fetchone()[0]
        
        if count > 0:
            conn.close()
            return
        
        initial_qa_pairs = [
            ("Write a Python function to check if a number is even", "def is_even(n):\n    return n % 2 == 0"),
            ("JavaScript function to reverse a string", "function reverse(str) {\n    return str.split('').reverse().join('');\n}"),
            ("Python function to find the maximum number in a list", "def find_max(numbers):\n    return max(numbers)"),
            ("JavaScript loop to print numbers 1 to 10", "for (let i = 1; i <= 10; i++) {\n    console.log(i);\n}"),
            ("Python function to calculate factorial of a number", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
            ("Node.js function to read a file", "const fs = require('fs');\nfunction readFile(filename) {\n    return fs.readFileSync(filename, 'utf8');\n}"),
            ("JavaScript function to check if a string is a palindrome", "function isPalindrome(str) {\n    const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, '');\n    return cleaned === cleaned.split('').reverse().join('');\n}"),
            ("Python function to count vowels in a string", "def count_vowels(text):\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in text if char in vowels)"),
            ("JavaScript function to get unique elements from an array", "function getUnique(arr) {\n    return [...new Set(arr)];\n}"),
            ("Python function to convert Celsius to Fahrenheit", "def celsius_to_fahrenheit(celsius):\n    return (celsius * 9/5) + 32"),
            ("Node.js function to write data to a file", "const fs = require('fs');\nfunction writeFile(filename, data) {\n    fs.writeFileSync(filename, data);\n}"),
            ("JavaScript function to sum all numbers in an array", "function sumArray(numbers) {\n    return numbers.reduce((sum, num) => sum + num, 0);\n}"),
            ("Python function to check if a string contains only digits", "def is_numeric(text):\n    return text.isdigit()"),
            ("JavaScript function to capitalize first letter of each word", "function capitalizeWords(str) {\n    return str.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');\n}"),
            ("Python function to remove duplicates from a list", "def remove_duplicates(lst):\n    return list(set(lst))"),
            ("Node.js simple HTTP server", "const http = require('http');\nconst server = http.createServer((req, res) => {\n    res.writeHead(200, {'Content-Type': 'text/plain'});\n    res.end('Hello World');\n});\nserver.listen(3000);"),
            ("JavaScript function to find the average of numbers in an array", "function average(numbers) {\n    return numbers.reduce((sum, num) => sum + num, 0) / numbers.length;\n}"),
            ("Python function to generate Fibonacci sequence up to n terms", "def fibonacci(n):\n    sequence = [0, 1]\n    for i in range(2, n):\n        sequence.append(sequence[i-1] + sequence[i-2])\n    return sequence[:n]"),
            ("JavaScript function to sort an array of objects by property", "function sortByProperty(arr, property) {\n    return arr.sort((a, b) => a[property] - b[property]);\n}"),
            ("Python function to calculate the area of a circle", "import math\ndef circle_area(radius):\n    return math.pi * radius ** 2"),
            ("JavaScript function to find minimum value in array", "function findMin(arr) {\n    return Math.min(...arr);\n}"),
            ("Python function to check if a number is prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
            ("Node.js function to make HTTP GET request", "const https = require('https');\nfunction httpGet(url, callback) {\n    https.get(url, (res) => {\n        let data = '';\n        res.on('data', (chunk) => data += chunk);\n        res.on('end', () => callback(data));\n    });\n}"),
            ("JavaScript function to convert string to title case", "function toTitleCase(str) {\n    return str.toLowerCase().split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');\n}"),
            ("Python function to count words in a string", "def count_words(text):\n    return len(text.split())"),
            ("JavaScript function to check if array contains a value", "function contains(arr, value) {\n    return arr.includes(value);\n}"),
            ("Python function to merge two lists", "def merge_lists(list1, list2):\n    return list1 + list2"),
            ("JavaScript function to get current timestamp", "function getCurrentTimestamp() {\n    return Date.now();\n}"),
            ("Python function to calculate square root", "import math\ndef square_root(n):\n    return math.sqrt(n)"),
            ("Node.js function to create directory", "const fs = require('fs');\nfunction createDir(dirName) {\n    if (!fs.existsSync(dirName)) {\n        fs.mkdirSync(dirName);\n    }\n}"),
            ("JavaScript function to format number with commas", "function formatNumber(num) {\n    return num.toString().replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');\n}"),
            ("Python function to convert string to lowercase", "def to_lower(text):\n    return text.lower()"),
            ("JavaScript function to generate random number between min and max", "function randomBetween(min, max) {\n    return Math.floor(Math.random() * (max - min + 1)) + min;\n}"),
            ("Python function to check if list is empty", "def is_empty(lst):\n    return len(lst) == 0"),
            ("JavaScript function to remove whitespace from string", "function trim(str) {\n    return str.trim();\n}"),
            ("Python function to get current date and time", "import datetime\ndef get_current_datetime():\n    return datetime.datetime.now()"),
            ("Node.js function to parse JSON from file", "const fs = require('fs');\nfunction parseJsonFile(filename) {\n    const data = fs.readFileSync(filename, 'utf8');\n    return JSON.parse(data);\n}"),
            ("JavaScript function to check if string starts with substring", "function startsWith(str, prefix) {\n    return str.startsWith(prefix);\n}"),
            ("Python function to calculate power of a number", "def power(base, exponent):\n    return base ** exponent"),
            ("JavaScript function to convert array to string", "function arrayToString(arr) {\n    return arr.join(',');\n}"),
            ("Python function to find length of string", "def string_length(text):\n    return len(text)"),
            ("JavaScript function to check if number is positive", "function isPositive(num) {\n    return num > 0;\n}"),
            ("Python function to create dictionary from two lists", "def create_dict(keys, values):\n    return dict(zip(keys, values))"),
            ("Node.js function to get file extension", "const path = require('path');\nfunction getFileExtension(filename) {\n    return path.extname(filename);\n}"),
            ("JavaScript function to round number to 2 decimal places", "function roundToTwo(num) {\n    return Math.round(num * 100) / 100;\n}"),
            ("Python function to sort list in ascending order", "def sort_ascending(lst):\n    return sorted(lst)"),
            ("JavaScript function to check if string ends with substring", "function endsWith(str, suffix) {\n    return str.endsWith(suffix);\n}"),
            ("Python function to get absolute value", "def absolute(n):\n    return abs(n)"),
            ("JavaScript function to convert string to number", "function toNumber(str) {\n    return Number(str);\n}"),
            ("Python function to check if key exists in dictionary", "def key_exists(dictionary, key):\n    return key in dictionary"),
            ("Node.js function to get current working directory", "const process = require('process');\nfunction getCurrentDir() {\n    return process.cwd();\n}"),
            ("JavaScript function to get length of array", "function arrayLength(arr) {\n    return arr.length;\n}"),
            ("Python function to replace substring in string", "def replace_substring(text, old, new):\n    return text.replace(old, new)"),
            ("JavaScript function to convert to uppercase", "function toUpper(str) {\n    return str.toUpperCase();\n}"),
            ("Python function to check if number is negative", "def is_negative(n):\n    return n < 0"),
            ("JavaScript function to get first element of array", "function getFirst(arr) {\n    return arr[0];\n}"),
            ("Python function to concatenate strings", "def concat_strings(str1, str2):\n    return str1 + str2"),
            ("Node.js function to check if file exists", "const fs = require('fs');\nfunction fileExists(filename) {\n    return fs.existsSync(filename);\n}"),
            ("JavaScript function to get last element of array", "function getLast(arr) {\n    return arr[arr.length - 1];\n}"),
            ("Python function to convert integer to string", "def int_to_string(n):\n    return str(n)"),
            ("JavaScript function to check if value is undefined", "function isUndefined(value) {\n    return value === undefined;\n}"),
            ("Python function to get keys from dictionary", "def get_keys(dictionary):\n    return list(dictionary.keys())"),
            ("JavaScript function to multiply two numbers", "function multiply(a, b) {\n    return a * b;\n}"),
            ("Python function to check if string is empty", "def is_empty_string(text):\n    return len(text) == 0"),
            ("Node.js function to join file paths", "const path = require('path');\nfunction joinPaths(path1, path2) {\n    return path.join(path1, path2);\n}"),
            ("JavaScript function to divide two numbers", "function divide(a, b) {\n    return b !== 0 ? a / b : null;\n}"),
            ("Python function to get values from dictionary", "def get_values(dictionary):\n    return list(dictionary.values())"),
            ("JavaScript function to add two numbers", "function add(a, b) {\n    return a + b;\n}"),
            ("Python function to convert list to string", "def list_to_string(lst):\n    return ' '.join(map(str, lst))"),
            ("JavaScript function to subtract two numbers", "function subtract(a, b) {\n    return a - b;\n}"),
            ("Python function to check if number is zero", "def is_zero(n):\n    return n == 0"),
            ("JavaScript function to check if array is empty", "function isEmptyArray(arr) {\n    return arr.length === 0;\n}"),
            ("Python function to get first element of list", "def get_first(lst):\n    return lst[0] if lst else None"),
            ("Node.js function to convert object to JSON string", "function objectToJson(obj) {\n    return JSON.stringify(obj);\n}"),
            ("Python function to solve Two Sum problem", "def two_sum(nums, target):\n    num_map = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in num_map:\n            return [num_map[complement], i]\n        num_map[num] = i\n    return []"),
            ("Python function to check if a string is a valid palindrome", "def is_valid_palindrome(s):\n    cleaned = ''.join(char.lower() for char in s if char.isalnum())\n    return cleaned == cleaned[::-1]"),
            ("Python function to find maximum subarray sum", "def max_subarray(nums):\n    max_sum = current_sum = nums[0]\n    for num in nums[1:]:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum"),
            ("Python function to merge two sorted arrays", "def merge_sorted_arrays(nums1, m, nums2, n):\n    i, j, k = m - 1, n - 1, m + n - 1\n    while j >= 0:\n        if i >= 0 and nums1[i] > nums2[j]:\n            nums1[k] = nums1[i]\n            i -= 1\n        else:\n            nums1[k] = nums2[j]\n            j -= 1\n        k -= 1"),
            ("Python function to remove duplicates from sorted array", "def remove_duplicates(nums):\n    if not nums:\n        return 0\n    i = 0\n    for j in range(1, len(nums)):\n        if nums[j] != nums[i]:\n            i += 1\n            nums[i] = nums[j]\n    return i + 1"),
            ("Python function to find single number in array", "def single_number(nums):\n    result = 0\n    for num in nums:\n        result ^= num\n    return result"),
            ("Python function to find intersection of two arrays", "def intersect(nums1, nums2):\n    from collections import Counter\n    count1, count2 = Counter(nums1), Counter(nums2)\n    result = []\n    for num in count1:\n        if num in count2:\n            result.extend([num] * min(count1[num], count2[num]))\n    return result"),
            ("Python function to move all zeros to end of array", "def move_zeroes(nums):\n    left = 0\n    for right in range(len(nums)):\n        if nums[right] != 0:\n            nums[left], nums[right] = nums[right], nums[left]\n            left += 1"),
            ("Python function to find majority element", "def majority_element(nums):\n    candidate, count = nums[0], 1\n    for i in range(1, len(nums)):\n        if nums[i] == candidate:\n            count += 1\n        else:\n            count -= 1\n            if count == 0:\n                candidate = nums[i]\n                count = 1\n    return candidate"),
            ("Python function to reverse integer", "def reverse_integer(x):\n    sign = -1 if x < 0 else 1\n    x = abs(x)\n    result = 0\n    while x:\n        result = result * 10 + x % 10\n        x //= 10\n    result *= sign\n    return result if -2**31 <= result <= 2**31 - 1 else 0"),
            ("Python function to find missing number in array", "def missing_number(nums):\n    n = len(nums)\n    expected_sum = n * (n + 1) // 2\n    actual_sum = sum(nums)\n    return expected_sum - actual_sum"),
            ("Python function to check if two strings are anagrams", "def is_anagram(s, t):\n    if len(s) != len(t):\n        return False\n    from collections import Counter\n    return Counter(s) == Counter(t)"),
            ("Python function to find first unique character in string", "def first_unique_char(s):\n    from collections import Counter\n    char_count = Counter(s)\n    for i, char in enumerate(s):\n        if char_count[char] == 1:\n            return i\n    return -1"),
            ("Python function to check if string contains all unique characters", "def is_unique(s):\n    return len(s) == len(set(s))"),
            ("Python function to rotate array to right by k steps", "def rotate_array(nums, k):\n    n = len(nums)\n    k = k % n\n    nums[:] = nums[-k:] + nums[:-k]"),
            ("Python function to find longest common prefix of strings", "def longest_common_prefix(strs):\n    if not strs:\n        return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix"),
            ("Python function to check if parentheses are valid", "def is_valid_parentheses(s):\n    stack = []\n    mapping = {')': '(', '}': '{', ']': '['}\n    for char in s:\n        if char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n        else:\n            stack.append(char)\n    return not stack"),
            ("Python function to find peak element in array", "def find_peak_element(nums):\n    left, right = 0, len(nums) - 1\n    while left < right:\n        mid = (left + right) // 2\n        if nums[mid] < nums[mid + 1]:\n            left = mid + 1\n        else:\n            right = mid\n    return left"),
            ("Python function to check if number is happy number", "def is_happy(n):\n    def get_sum(num):\n        total = 0\n        while num:\n            digit = num % 10\n            total += digit * digit\n            num //= 10\n        return total\n    \n    seen = set()\n    while n != 1 and n not in seen:\n        seen.add(n)\n        n = get_sum(n)\n    return n == 1")
        ]
        
        cursor.executemany('INSERT INTO qa_pairs (question, answer) VALUES (?, ?)', initial_qa_pairs)
        conn.commit()
        conn.close()
        print("Database populated with initial Q&A pairs!")
    
    def save_qa(self, question, answer):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO qa_pairs (question, answer) VALUES (?, ?)
        ''', (question, answer))
        
        conn.commit()
        conn.close()
    
    def get_answer(self, question):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT answer FROM qa_pairs WHERE question = ? ORDER BY timestamp DESC LIMIT 1
        ''', (question,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_all_qa(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT question, answer, timestamp FROM qa_pairs ORDER BY timestamp DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def search_questions(self, keyword):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT question, answer FROM qa_pairs 
            WHERE question LIKE ? ORDER BY timestamp DESC
        ''', (f'%{keyword}%',))
        
        results = cursor.fetchall()
        conn.close()
        
        return results