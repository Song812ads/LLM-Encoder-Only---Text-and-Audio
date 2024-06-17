import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer
from tensorflow.keras.models import load_model
import re
# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('minhtoan/vietnamese-gpt2-finetune')

text = "Hôm nay ngày 1. Hãy mở đèn 1 "
clean_data = re.sub(r'[^\w\s]', '', text).lower().strip()
# print(clean_data)
input_ids = tokenizer.encode(clean_data)
print(tokenizer.decode(input_ids))
# print(input_ids)
# print(input_ids)
# padded_tokens = np.pad(input_ids, (0, max(0, 30 - len(input_ids))), mode='constant', constant_values=np.random.randint(0, 9194, 1))
# print([0]*30)
input_ids += [0] * (30 - len(input_ids))
word_vector = np.array(input_ids).reshape(30,)

# Load TensorFlow Lite interpreter
interpreter = tf.lite.Interpreter(model_path="model_tf/text_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input tensor (assuming single input tensor)
interpreter.set_tensor(input_details[0]['index'], np.array([word_vector], dtype=np.float32))  # Keep integers

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Process the output data as needed
print(f'Test: {text}, result class: {(output_data)}')
