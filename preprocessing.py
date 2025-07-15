import os
import json
import re
from typing import List

import fitz  # PyMuPDF for PDF parsing
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever


import pdfplumber
import json
import re
from collections import defaultdict

# Load environment variables from .env
load_dotenv()

# Constants
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "RAG_DOCS"
CHROMA_DB_PATH = "chromaDB/saved/"
PDF_DIR = "PDFs"
ALL_DOCS_JSON = "all_docs.json"


def recursive_split(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text using character-level recursion based on newlines and punctuation.
    
    Args:
        text: The raw text to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between adjacent chunks.
    
    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def semantic_chunker(text: str, embeddings_model) -> List[str]:
    """
    Split the text into semantically meaningful chunks using a semantic chunker.

    Args:
        text: Text to split.
        embeddings_model: Pre-loaded embedding model.

    Returns:
        List of semantically coherent text chunks.
    """
    chunks = []
    for segment in recursive_split(text):
        chunker = SemanticChunker(embeddings_model)
        chunks.extend(chunker.split_text(segment))
    return chunks

def heading_above(table_obj, page):
    """
    Find the single text line whose bottom edge is closest above the table,
    but skip any line that contains digits, '$' or '%', since those are
    usually units/column headers, not the true title.
    """
    x0, top, x1, _ = table_obj.bbox
    words = page.extract_words(use_text_flow=True)

    # cluster words into lines by their top y-coordinate
    lines = defaultdict(list)
    for w in words:
        key = round(w["top"], 0)
        lines[key].append(w)

    # build (y_bottom, text) candidates for any line that overlaps horizontally
    candidates = []
    for y0, ws in lines.items():
        y1 = max(w["bottom"] for w in ws)
        if y1 < top and any(w["x0"] < x1 and w["x1"] > x0 for w in ws):
            text = " ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"]))
            candidates.append((y1, text))

    if not candidates:
        return ""

    # filter out any line with digits, $ or %
    filtered = [(y, t) for (y, t) in candidates if not re.search(r"[\d\$\%]", t)]

    # use the filtered list if non-empty, otherwise fall back to all candidates
    use = filtered if filtered else candidates

    #  pick the one whose bottom edge is highest (closest to the table)
    return max(use, key=lambda t: t[0])[1]

def extract_tables_with_headings(
    pdf_path: str
) -> List[Document]:
    
    docs: List[Document] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for table_no, table_obj in enumerate(page.find_tables(), start=1):

                try:
                    # this is where ZeroDivisionError is coming from
                    table_data = table_obj.extract()
                except ZeroDivisionError:
                    # skip any table that blows up
                    continue

                if not table_data:
                    continue
               
                title      = heading_above(table_obj, page) or f"Table {table_no}"
                
                table_lines = ["\t".join(cell or "" for cell in row) for row in table_data]
                table_text  = "\n".join(table_lines)

                # build Document
                doc = Document(
                    page_content=table_text,
                    metadata={
                        "page":     page_num,
                        "table_no": table_no,
                        "heading":  title
                    }
                )
                docs.append(doc)
    
    return docs


def extract_chunks_from_pdf(pdf_path: str, embeddings_model) -> List[Document]:
    """
    Extract semantic chunks from a given PDF.

    Args:
        pdf_path: Path to the PDF file.
        embeddings_model: Embedding model for semantic chunking.

    Returns:
        List of Document objects containing chunked text and metadata.
    """
    documents = []
    try:
        print(f'🗂️  Getting PDF from: {pdf_path}\n')
        pdf = fitz.open(pdf_path)

        for page_idx, page in enumerate(pdf):
            print(f'📖 Reading Page no: {page_idx + 1}')
            raw_text = re.sub(r"\s+", " ", page.get_text("text")).strip()

            if not raw_text:
                continue

            for idx, chunk in enumerate(semantic_chunker(raw_text, embeddings_model)):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "page": page_idx + 1,
                            "chunk": idx,
                            "source": os.path.basename(pdf_path)
                        }
                    )
                )
        pdf.close()
        table_docs = extract_tables_with_headings(pdf_path)
        documents.extend(table_docs)


    except Exception as e:
        print(f"❌ Error while processing PDF '{pdf_path}': {e}")

    return documents


def save_docs(docs: List[Document], filepath: str = ALL_DOCS_JSON) -> None:
    """
    Save documents to JSON file.

    Args:
        docs: List of Document objects to save.
        filepath: Destination file path.
    """
    try:
        print("📥📄 Saving chunks for future use...")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
                f,
                indent=2
            )
    except Exception as e:
        print(f"❌ Error saving documents: {e}")


def load_docs(filepath: str = ALL_DOCS_JSON) -> List[Document]:
    """
    Load documents from a previously saved JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        List of Document objects.
    """
    try:
        print("📤📄 Loading chunks...")
        if not os.path.exists(filepath):
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in data
        ]
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return []


def init_chroma() -> Chroma:
    """
    Initialize or load a Chroma vector store.

    Returns:
        A Chroma object ready for indexing or retrieval.
    """
    print('🧭 Creating or loading ChromaDB...')
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    )


def create_chunks(PDF_FILES: List[str], PDF_DIR: str = 'PDFs') -> None:
    """
    Process PDF files and generate semantic chunks for RAG.

    Args:
        PDF_FILES: List of PDF filenames (without extension).
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    docs = load_docs(ALL_DOCS_JSON)
    PDF_PATH = PDF_DIR +'/'
    if not docs:
        for name in PDF_FILES:
            path = os.path.join(PDF_PATH, f"{name}.pdf")
            if os.path.exists(path):
                docs.extend(extract_chunks_from_pdf(path, embeddings))
            else:
                print(f"⚠️  File not found: {path}")
        save_docs(docs)
    else:
        print("✅ Pre-saved chunks loaded. Skipping PDF parsing.")


def hybrid_search(query: str) -> List[Document]:
    """
    Perform hybrid search using both BM25 and Chroma vector retrievers.

    Args:
        query: The user query.

    Returns:
        List of retrieved Document objects.
    """
    try:
        chroma_store = init_chroma()

        # Load and index documents if not already indexed
        if chroma_store._collection.count() == 0:
            docs = load_docs(ALL_DOCS_JSON)
            if docs:
                chroma_store.add_documents(docs)
            else:
                print("⚠️  No documents available for retrieval.")
                return []

        # Load for BM25
        docs = load_docs(ALL_DOCS_JSON)
        if not docs:
            return []

        chroma_retriever = chroma_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20}
        )

        bm25_retriever = BM25Retriever.from_texts(
            [d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
            k=5
        )

        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[1, 1]
        )

        return ensemble.invoke(query)

    except Exception as e:
        print(f"❌ Error in hybrid search: {e}")
        return []
