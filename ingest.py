import os
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

DOCS_PATH = "docs"
VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "BAAI/bge-m3"

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
]


def clean_text(text):
    text = text.replace("\\`", "'")
    text = re.sub(r"(?m)^\\-", "-", text)
    text = re.sub(r"(?m)[ \t]+$", "", text)
    return text


def build_section(metadata, fallback):
    titles = []
    for key in ("h1", "h2", "h3", "h4"):
        title = metadata.get(key, "").strip()
        if title and (not titles or titles[-1].lower() != title.lower()):
            titles.append(title)
    return " > ".join(titles) if titles else fallback


loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,
)

documents = loader.load()
print(f"Loaded documents: {len(documents)}")

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON,
    strip_headers=True,
)

char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = []
for doc in documents:
    fallback = os.path.splitext(os.path.basename(doc.metadata["source"]))[0]
    sections = header_splitter.split_text(clean_text(doc.page_content))

    for chunk in char_splitter.split_documents(sections):
        body = chunk.page_content.strip()
        if not body:
            continue

        section = build_section(chunk.metadata, fallback)
        chunk.metadata = {"section": section}
        chunk.page_content = f"{section}\n\n{body}"
        chunks.append(chunk)

print(f"Chunks created: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)

vectorstore.save_local(VECTORSTORE_PATH)

print("Vector store saved.")