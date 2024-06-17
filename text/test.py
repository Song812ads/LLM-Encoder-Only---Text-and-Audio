import nltk
from nltk.tokenize import word_tokenize
import re
# You may need to download the Punkt tokenizer models if not already downloaded
nltk.download('punkt')

text = "Xin chào, tôi là một câu hỏi đơn giản."
clean_data = re.sub(r'[^\w\s]', '', text).lower().strip()
words = clean_data.split()
vocabulary = list(set(words))
print(words)
print(vocabulary)