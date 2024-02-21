from src.cot_utils import *
from src.dataset_utils import *

# Bloomz model
model_name = "bigscience/bloomz-7b1-mt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# inputstring = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

# inputs = tokenizer.encode(inputstring, return_tensors="pt")
# outputs = model.generate(inputs)

# print(tokenizer.decode(outputs[0]))

# Create a text generation pipeline
text_generator = pipeline("text-generation", model=model_name)

# Define the input string
input_string = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

# Generate output using the pipeline
outputs = text_generator(input_string)

# Print the generated text
print(outputs[0]['generated_text'])