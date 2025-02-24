from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import torch
import io
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# Load the sentence transformer model for retrieval
retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load a FREE Hugging Face LLM (Ensuring it fits within token limits)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
pipe = pipeline("text-generation", model=llm, tokenizer=tokenizer)

class Query(BaseModel):
    question: str

def extract_text_from_pdf(pdf_file: UploadFile):
    """Extracts text from an uploaded PDF file."""
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_file.file.read())) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def chunk_text(text, max_tokens=500):
    """Splits text into smaller chunks based on word count (approximate token count)."""
    words = text.split()
    chunks = []
    while words:
        chunk = words[:max_tokens]
        chunks.append(" ".join(chunk))
        words = words[max_tokens:]
    return chunks

@app.post("/query/")
async def process_query(
    question: str = Form(...),  
    files: List[UploadFile] = File(...)
):
    """Processes a query based on multiple uploaded PDF files."""
    try:
        all_text = ""

        for file in files:
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload PDFs only.")
            all_text += extract_text_from_pdf(file) + "\n\n"

        if not all_text:
            raise HTTPException(status_code=400, detail="No text extracted from PDFs.")

        # ✅ Split text into smaller sections
        text_chunks = chunk_text(all_text, max_tokens=500)

        # ✅ Compute embeddings for query and text chunks
        query_embedding = retrieval_model.encode(question, convert_to_tensor=True)
        chunk_embeddings = retrieval_model.encode(text_chunks, convert_to_tensor=True)

        # ✅ Compute similarity scores & select top relevant sections
        similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
        top_indices = torch.topk(similarities[0], k=min(3, len(text_chunks))).indices.tolist()
        relevant_chunks = [text_chunks[i] for i in top_indices]

        # ✅ Use LLM for answering (process each chunk separately)
        responses = []
        for chunk in relevant_chunks:
            input_text = f"Based on the following text, answer the question: {question}\n\n{chunk}"
            response = pipe(input_text, max_new_tokens=150)[0]["generated_text"]
            responses.append(response.strip())

        # ✅ Combine responses & return
        return {"answer": " ".join(responses)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# **************************************SOLUTION 2************************************************************************************
# from typing import List
# from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from fastapi import Form
# import openai
# import pdfplumber
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# import torch
# import io

# client = openai.OpenAI(api_key="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") 
# app = FastAPI()

# # Load the sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins (for testing; restrict in production)
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )

# def extract_text_from_pdf(pdf_file: UploadFile):
#     """Extract text from an uploaded PDF file."""
#     try:
#         text = ""
#         with pdfplumber.open(io.BytesIO(pdf_file.file.read())) as pdf:
#             for page in pdf.pages:
#                 text += page.extract_text() + "\n"
#         return text.strip()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# class Query(BaseModel):
#     question: str

# @app.post("/query/")
# async def process_query(
#     question: str = Form(...),  # Accept query as a form field
#     files: List[UploadFile] = File(...)
# ):
#     """Process a query based on multiple uploaded PDF files."""
#     try:
#         all_text = ""  # Store combined text from all PDFs

#         for file in files:
#             if file.content_type != "application/pdf":
#                 raise HTTPException(status_code=400, detail="Invalid file type. Please upload PDFs only.")
#             all_text += extract_text_from_pdf(file) + "\n\n"

#         # ✅ Split text into smaller sections for better retrieval
#         sections = all_text.split("\n\n")

#         # ✅ Compute embeddings for query and document sections
#         query_embedding = model.encode(question, convert_to_tensor=True)
#         section_embeddings = model.encode(sections, convert_to_tensor=True)

#         # ✅ Compute similarity scores
#         similarities = util.pytorch_cos_sim(query_embedding, section_embeddings)
#         top_indices = torch.topk(similarities[0], k=3).indices.tolist()
#         relevant_sections = [sections[i] for i in top_indices]

#         # ✅ Use OpenAI's new API format
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",  # or "gpt-4" if you have access
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": f"Based on the following document sections, answer the question: {question}\n\nSections:\n{relevant_sections}"}
#             ],
#             max_tokens=150
#         )

#         # ✅ Corrected response extraction
#         return {"answer": response.choices[0].message.content.strip()}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
 
# **********************************SOLUTION 1*******************************************************************************************    
   
# async def process_query(query: Query, files: List[UploadFile] = File(...)):
#     """Process a query based on multiple uploaded PDF files."""
#     try:
#         all_text = ""  # Store combined text from both PDFs
        
#         for file in files:
#             if file.content_type != "application/pdf":
#                 raise HTTPException(status_code=400, detail="Invalid file type. Please upload PDFs only.")
#             all_text += extract_text_from_pdf(file) + "\n\n"

#         # Split text into smaller sections (optional, for better retrieval)
#         sections = all_text.split("\n\n")  # Split into paragraphs
        
#         # Compute embeddings for query and document sections
#         query_embedding = model.encode(query.question, convert_to_tensor=True)
#         section_embeddings = model.encode(sections, convert_to_tensor=True)
        
#         # Compute similarity scores
#         similarities = util.pytorch_cos_sim(query_embedding, section_embeddings)
#         top_indices = torch.topk(similarities[0], k=3).indices.tolist()
#         relevant_sections = [sections[i] for i in top_indices]

#         # Generate a response using OpenAI
#         response = openai.Completion.create(
#             engine="text-davinci-003",
#             prompt=f"Based on the following document sections, answer the question: {query.question}\n\nSections:\n{relevant_sections}",
#             max_tokens=150
#         )
#         return {"answer": response.choices[0].text.strip()}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
