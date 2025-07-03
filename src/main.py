from google.colab import files
from src.chunker import get_clean_text, split_text_to_chunks
from src.faiss_db import add_chunks_to_index, search_query

# Upload DOCX
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Chunking
text = get_clean_text(file_name)
chunks = split_text_to_chunks(text)

# Indexing
add_chunks_to_index("ITU_assignment_word_format.docx", chunks)

# Search
query = input("Enter your question: ")
search_query(query, top_k=3)
