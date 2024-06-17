from gensim.models import Word2Vec
import underthesea #thu vien phan tich tieng viet
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from pyvi import ViTokenizer
# import nltk
# from nltk.tokenize import word_tokenize

# nltk.download('punkt')

w2v_model = Word2Vec.load('model_w2v/w2v_model.hdf5')
test_data = "Hôm nay thứ 3. Hãy mở đèn 1 "
clean_data = re.sub(r'[^\w\s]', '', test_data).lower().strip()
test_token = clean_data.split()
word_vector_test = []

# Iterate through each word in the vocabulary
for word in test_token:
    # Append word vectors to the list
    if word in w2v_model.wv.key_to_index: 
        print(word)
        word_vector_test.append(w2v_model.wv[word])

# Pad the word vectors with zeros if the length is less than 8
word_vector_test += [[0] * 300] * (20 - len(word_vector_test))
word_vectors = np.array(word_vector_test).reshape(20, 300)

interpreter = tf.lite.Interpreter(model_path="model_w2v/text_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data (this is just a placeholder, replace it with your actual input data)
input_data = ...  # Your input data in the appropriate format

# Set input tensor (assuming single input tensor)
interpreter.set_tensor(input_details[0]['index'], np.array([word_vectors],dtype=np.float32))

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Process the output data as needed
print(f'Test: {test_data}, result class: {(output_data)}')
# print(np.argmax(output_data))
# print(model(word_vectors))