!pip install faiss-cpu
!pip install sentence-transformers
!pip install gradio

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr

model = SentenceTransformer('all-mpnet-base-v2')

def get_embeddings(texts):
  embeddings = model.encode(texts)
  return embeddings

def process_text(text_input, text_prompt):
    text_input = [text.strip() for text in text_input.split(",")]

    input_embeddings = get_embeddings(text_input)

    d = input_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(input_embeddings).astype('float32'))

    prompt_embedding = get_embeddings([text_prompt])[0]

    D, I = index.search(np.array([prompt_embedding]).astype('float32'), k=1)

    answer = text_input[I[0][0]]

    embeddings_output = ""
    for i, embedding in enumerate(input_embeddings):
      embeddings_output += f"Text {i+1}: {embedding}\n"

    embeddings_output += f"\nPrompt Embedding: {prompt_embedding}\n"
    embeddings_output += f"\nNearest neighbor index: {I[0][0]}\n"
    embeddings_output += f"Distance: {D[0][0]}\n"

    return embeddings_output, answer


iface = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(label="Enter text (separate multiple texts with commas)"),
        gr.Textbox(label="Enter text prompt")
    ],
    outputs=[
        gr.Textbox(label="Embeddings"),
        gr.Textbox(label="Answer")
    ],
    title="Vector Database Demonstration",
    description="Enter text, create embeddings with Faiss, then ask a question about the text. Brought to you by Uma, Aishwarya, Prahasya and Bhavana from the CSE Deptartment, GNITS Hyderabad."

)


iface.launch()

