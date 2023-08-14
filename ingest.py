"""This is the logic for ingesting Notion data into LangChain."""
import time
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle


# Here we load in the data in the format that Notion exports it in.
ps = list(Path("data/").glob("*.txt"))

data = []
sources = []
for p in ps:
    with open(p, encoding="utf-8") as f:
        data.append(f.read())
    sources.append(p)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
docs = []
metadatas = []
store = None
for i, d in enumerate(data):
    print("step", i + 1, "of", len(data))
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))
    if store is None:
        store = FAISS.from_texts(splits, OpenAIEmbeddings(), metadatas=[{"source": sources[i]}] * len(splits))
    else:
        store.add_texts(splits, metadatas=[{"source": sources[i]}] * len(splits))
    faiss.write_index(store.index, "docs.index")
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
    print("step", i, "of", len(data), "done")
    # if i % 3 == 2:
    time.sleep(60)


# Here we create a vector store from the documents and save it to disk.
# store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
