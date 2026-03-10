#required libraries
#1.transfformers: for loading pre-trained transformer models
#2 torch :The backend for ml models , tensor computations and gpu acceleration
#3.SentenceTransformer: for creating high quality text embeddings
#4.pypdf: to read and extract text from pdf files
#5.Faiss-cpu: library from meta ai for efficient similarity search in our vector store 
# FAISS = Facebook AI similarity search used for fast similarity search between vectors 
# faiss uses cosine similarity and euclidean distance to find similar vectors in high dimensional space

#step 1:Load and chunk the document
#LLM has limited context window(the amount of text they can look at once 
# we cannot feed it a 100 page pdf . so we will load the pdf and split it text into small, overlapping chunks 
import pypdf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
#import numpy as np
import faiss
# document loading and chunking
def load_and_chunk_pdf(file_path, chunk_size=512, overlap=50):
    #overlapping prevents losing of context between chunks 
    # loads pdf extracts text and split into chunks
    print("loading and chunking the pdf")
    reader=pypdf.PdfReader(file_path)# loads the pdf using the librarypypdf and contains all pages of pdf
    text=""
    for page in reader.pages:# go through every page in pdf and extract text from each page and astore it in text variable
        text+=page.extract_text() or ""
    # whole document matter will be in the text
    #simple chunking tools
    chunks=[]
    for i in range(0,len(text),chunk_size-overlap):#each chunk overalps previos chunk by 50characters
        chunks.append(text[i:i +chunk_size])
    print("created chunks")
    return chunks
    
pdf_chunks=load_and_chunk_pdf(r"C:\Users\veera\Downloads\OneDrive\Documents\Sankalp\ML practice\agentic ai\huggingface\the-illusion-of-thinking.pdf")
embedding_model=SentenceTransformer("all-MiniLM-L6-v2")  
#Step2: embed chunks and store in vector database model
print("load embedding model")
model=SentenceTransformer("all-MiniLM-L6-v2")

#create embeddings and vector store(FAISS)
def create_vector_store(chunks, model):
    # embeds chunks and store them in FAISS vector store
    print("creating embedding and vector store ")
    embeddings=model.encode(chunks, convert_to_tensor=True)
    #returns embeddings as pytorch tensor (instead of numpy) enabling gpu accerleration and compatibility with deeplearning operations
    embeddings_np=embeddings.cpu().numpy()#moves tensor from gpu to cpu and converts it to numpy array for faiss
    #this necessary because faiss works with numpy arrays 
    
    #faiss index setup
    dimension=embeddings_np.shape[1]
    index=faiss.IndexFlatL2(dimension)# creates faiss index using euclidean distance 
    #stores embeddings in a flat structure without compression and uses l2 distance for similarity search
    index.add(embeddings_np)
    #add all numpy embeddings into faiss index . this index shows all 384 dimensional vectors from pdf chunks and can perform fast nearst neighbour searches
    print("vector store created successfully")
    return index

vector_store=create_vector_store(pdf_chunks, model)

#Step 3: Retrieve relevant chunks and generate answer
def retrieve_relevant_chunks(query, model, chunks, index, top_k=3):
    #finds most relevant chunk for given query
    print(f"retrieving top {top_k} chunks for query: {query}")
    query_embedding=model.encode([query], convert_to_tensor=True).cpu().numpy()
    #search the faiss index
    
    distances, indices=index.search(query_embedding, top_k)
    
    #return actual text chunks
    return [chunks[i] for i in indices[0]]

# step 4 augment prompt and genarate answer
#generate logic

print("loading language model")
# using smaller , manegeable model for local demonstration
llm_tokenizer=AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_answer(query, context_chunks):
    #builds a prompt and generate answer using the LLM
    print("generating answer using LLM")
    context=" ".join(context_chunks)
    prompt=f"""please answer the following question based only on the provided context.
    If context does not contain the answer say I donot have enough informarion to answer this."
    Context: {context}
    Question: {query}
    Answer:"""
    
    inputs=llm_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs=llm_model.generate(**inputs, max_length=200, temperature=0.1, top_p=0.95)
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

#Main chatbot logic

def ask_chatbot(query):
    # function to chat with pdf 
    retrieved_chunks=retrieve_relevant_chunks(query, embedding_model, pdf_chunks, vector_store)
    answer=generate_answer(query, retrieved_chunks)
    
    return answer


user_question=" what are the main tpics discussed in the pdf?"

final_answer= ask_chatbot(user_question)
print("Answer:", final_answer) 