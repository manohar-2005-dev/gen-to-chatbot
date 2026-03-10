'''
we will use huggung face ecosysytem which is industry standard for deploying and experimenting genai models .
Specifically we use the transformers library and a small but capable LLM to get our feet wet with text generation .

you will need to install python and hugging face transformers library installed

pip install transformers torch(pytorch is backend for transformers and is required for most models to run)

A deep learning framework
Used to build and train neural networks
Required backend for most transformer models
we will use distil GPT-2 smaller and faster version of gpt2 for quick execution
'''
# Initialize text generation pipeline
# input text to  tokens to vector embeddings to input and passed through neural network models ,find the next word by greedy decoding 
#top p and top k sampling is used and the word which is choosed is send as input for next word prediction and this process continues until the end of sentence token is generated
#this iterative process is called autoregressive generation
#the model generates text one token at a time, using the previously generated tokens as context for predicting the next token.
from transformers import pipeline
# Initialize the text generation pipeline with the specified model
generator=pipeline("text-generation",model="distilgpt2")
#define prompt
prompt="The AI agent quickly analyzed the data stream, and its core decision was to"
# generate text with sampling parameters
# we will use parameters to make the output more creative and less repetitive
output= generator(prompt,
                  max_length=100,#stop after 100 tokens 
                  num_return_sequences=1,#generate one sequence
                  do_sample=True,#enable sampling for more creative output
                  top_k=50,#consider the top 50 tokens for sampling
                  top_p=0.95,#consider tokens with cumulative probability of 0.95
                  temperature=0.75,#lower temperature makes output more focused and deterministic, while higher temperature increases randomness and creativity
                  repetition_penalty=1.2,#penalty for repeating the same token, higher value discourages repetition
                  no_repeat_ngram_size=2,#prevent repeating the same bigram (2-gram) to reduce redundancy
                  )
# the above returns list of generated sequences, we will extract the first one, inside list there is a dictionary with key 'generated_text' which contains the generated text
#print result
print("Prompt")
print(prompt)
print("\nGenerated Text")
print(output[0]['generated_text'])