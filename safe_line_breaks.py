import os
import glob
import re

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find Korean character + period + space, and replace with Korean character + period + \n\n
    # We must insert the literal characters '\n\n' so that Python parses it as newlines in strings,
    # rather than actual line breaks in the source code which break single-quote strings.
    new_content = re.sub(r'([가-힣])\.\s+', r'\1.\\n\\n', content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")

files = glob.glob('f:/app/3. market 코스피 전망/**/*.py', recursive=True)
for file in files:
    if '.venv' in file or '__pycache__' in file or file.endswith('safe_line_breaks.py'):
        continue
    process_file(file)

print("Line breaks applied safely.")
