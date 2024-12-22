# Matrix Applications in NLP using Vector Databases - A Demonstration.

## Installation of Required Libraries
```python
!pip install faiss-cpu
!pip install sentence-transformers
```
#### •  !pip install:  Installs external libraries in a Jupyter Notebook or similar Python environment.
#### • faiss-cpu: Library for similarity search and clustering of dense vectors.
#### • sentence-transformers: Provides pre-trained models for converting sentences into vector embeddings. 


## Import Libraries
```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
```
#### •  faiss: A library to perform fast similarity searches among vectors.
#### •  numpy: Used for numerical operations such as creating arrays and performing mathematical computations.
#### •  SentenceTransformer: A tool for encoding text into dense numerical vector representations.


## Install Gradio
```python
!pip install gradio
import gradio as gr
```
#### •  gradio: A library to create simple and interactive GUIs for Python functions.


## Initialize SentenceTransformer Model
```python
model = SentenceTransformer('all-mpnet-base-v2')
```
#### •  Loads the all-mpnet-base-v2 model, which converts input sentences into dense embeddings (numerical vectors).


## Define the get_embeddings Function
```python
def get_embeddings(texts):
  embeddings = model.encode(texts)
  return embeddings
```
#### •  Purpose: Converts a list of texts into dense embeddings.
#### •  model.encode(texts): Encodes the input texts into numerical vectors.


## Define the process_text Function
```python
def process_text(text_input, text_prompt):
    text_input = [text.strip() for text in text_input.split(",")]
```
#### •  Input: A string of texts separated by commas.
#### •  split(","): Splits the string into a list of individual texts.
#### •  strip(): Removes any leading or trailing spaces from each text.


## Generate Embeddings for Input Texts
```python
   input_embeddings = get_embeddings(text_input)
```
#### •  Calls the get_embeddings function to create embeddings for the input texts.
#### •  input_embeddings: A 2D array where each row represents the embedding of one text.


## Create a FAISS Index
```python
   d = input_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(input_embeddings).astype('float32'))
```
#### •  d: The number of dimensions in the embeddings (columns in the input_embeddings array).
#### •  IndexFlatL2(d): Initializes a FAISS index for similarity search using L2 distance (Euclidean distance).
#### •  index.add(): Adds the input embeddings to the FAISS index after converting them to 32-bit floating-point format.


## Generate Prompt Embedding
```python
   prompt_embedding = get_embeddings([text_prompt])[0]
```
#### •  Converts the input prompt (text_prompt) into an embedding.
#### •  The embedding is stored as prompt_embedding.


## Perform Similarity Search
```python
   D, I = index.search(np.array([prompt_embedding]).astype('float32'), k=1)
```
#### •  index.search(): Searches for the top k=1 nearest neighbors to the prompt_embedding in the FAISS index.
#### •  D: A list of distances to the nearest neighbors.
#### •  I: A list of indices corresponding to the nearest neighbors.


## Retrieve the Nearest Neighbor
```python
   answer = text_input[I[0][0]]
```
#### •  I[0][0]: Index of the nearest neighbor in the input texts.
#### •  text_input[I[0][0]]: Retrieves the nearest neighbor's text from the text_input list.


## Create the Embeddings Output
```python
   embeddings_output = ""
    for i, embedding in enumerate(input_embeddings):
      embeddings_output += f"Text {i+1}: {embedding}\n"


    embeddings_output += f"\nPrompt Embedding: {prompt_embedding}\n"
    embeddings_output += f"\nNearest neighbor index: {I[0][0]}\n"
    embeddings_output += f"Distance: {D[0][0]}\n"
```
#### •  Loops through the input_embeddings and appends each text's embedding to embeddings_output.
#### •  Also appends the prompt_embedding, nearest neighbor's index, and the distance.


## Return Results
```python
   return embeddings_output, answer
```
#### •  Returns the embeddings output and the nearest neighbor's text as the "answer."


## Create the Gradio Interface
```python
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
    title="Text Similarity and Question Answering",
    description="Enter text, create embeddings with Faiss, then ask a question about the text."
)
```
#### •  gr.Interface(): Creates a GUI for the process_text function.

### •  Inputs:
#####   1)  A textbox for entering texts (comma-separated).
#####   2)  A textbox for the prompt.

### •  Outputs:
#####   1)  One textbox for the embeddings.
#####   2)  Another textbox for the nearest neighbor (answer).

#### •  Title and Description: Provides context for the GUI.


## Launch the Interface
```python
iface.launch()
```
#### •  Launches the Gradio interface, making it accessible to the user in a browser or notebook.


## Summary
#### 1)  Takes input texts and a query prompt.
#### 2)  Converts the texts and prompt into dense embeddings using SentenceTransformer.
#### 3)  Creates a FAISS index to store and search among embeddings.
#### 4)  Finds the text most similar to the prompt using similarity search.
#### 5)  Displays the embeddings and the closest text in a user-friendly GUI built with Gradio.

