#Sentence transformers library (built on pytorch and hugging face transformers) is the industry standard toolkit for generating high quality dense sentence and paragragph embeddings 

#Algorithms steps that we use in this 
#1. Choose an embedding model which is optimized for sentence level tasks
#2 the model internally tokenizes the input sentence
#3. the tokens pass through transformer model
#4.pooling: since transformers  ouput an embedding vector for each token we need one vector 
# for entire sentnece.pooling aggregates all token vectors into a single fixed size sentence vector
#.5.Normalization: the final vector which is formed by pooling is usually normalized(itd lenght is set to 1) 
#so that vector similarity (like cosine similarity) only depends on the angle between the vectors not their magnitude

#importing the libraryies
from sentence_transformers import SentenceTransformer
#import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#cosine similarity measures angle between two vectors in a high dimensional space 
#find how similar two vectors 

#model selection and loading 
#'all-MiniLM-L6-v2' is a light weight sentence  transformer model trained by hugging face team

print("Loading the sentence transformr model...")
model=SentenceTransformer('all-MiniLM-L6-v2')
print("model loaded successfully")

# Defining sentences to encode
sentences=[
    "A dog is chasing a bright red ball on a sunny green field. ",
    "The canine pursues a crimson sphere across the lawn.",
    "I'm planning to bake an apple pie for the picnic next weekend.",
    "A new laptop was purchased from tech store today."
]

#genrating sentence embeddings
print("Generating sentence embeddings...")
embeddings=model.encode(sentences)

# now we will look at shape and size of vectors
print(f"Shape of embeddings:{embeddings.shape}")#output will be (4,384)
# here 4 indicates number of sentences and 384 represents the size of embedding vector for each vector

print(f"embedding dimension:{embeddings.shape[1]}")
# here each sentence has 384 dimensional vector ,
#each dimension captures different semantic feature of sentence
# here dimension of vector is number of features in vector

print(f"sample of embedding vector for first sentence:\n{embeddings[0]}")

#calculate the semantic similarity using cosine similarity

#now we will compare first sentence s1 with others 
#calculate the cosine similarity matrix between all pairs
similarity_matrix=cosine_similarity(embeddings)

#extracting and displaying key similarities

S1_S2_sim = similarity_matrix[0,1]
S1_S3_sim = similarity_matrix[0,2]
S1_S4_sim = similarity_matrix[0,3]

print(f"similarity between sentence 1 and sentence 2: {S1_S2_sim:.4f}")
print(f"similarity between sentence 1 and sentence 3: {S1_S3_sim:.4f}")
print(f"similarity between sentence 1 and sentence 4: {S1_S4_sim:.4f}")
