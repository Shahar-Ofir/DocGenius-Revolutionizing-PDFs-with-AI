"""PDF question answering, kept free of Streamlit so it can be evaluated in CI."""
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# text-davinci-003 (langchain's default) is retired; this is the remaining completions model.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-instruct")
REQUEST_TIMEOUT_S = 60


def build_knowledge_base(pdf_file):
    """Index a PDF (path or file-like). Each chunk keeps its 1-based page number."""
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks, metadatas = [], []
    for page_no, page in enumerate(PdfReader(pdf_file).pages, start=1):
        for chunk in splitter.split_text(page.extract_text() or ""):
            chunks.append(chunk)
            metadatas.append({"page": page_no})
    if not chunks:
        raise ValueError("No extractable text found in the PDF.")
    embeddings = OpenAIEmbeddings(request_timeout=REQUEST_TIMEOUT_S, max_retries=3)
    return FAISS.from_texts(chunks, embeddings, metadatas=metadatas)


def answer_question(knowledge_base, question, temperature=0.7):
    """Return (answer, source_docs). source_docs are the chunks the answer was based on,
    most relevant first; each has .page_content and .metadata["page"]."""
    docs = knowledge_base.similarity_search(question)
    # Timeout, output-token ceiling and bounded retries (read-only call, safe to retry)
    llm = OpenAI(
        model_name=OPENAI_MODEL,
        temperature=temperature,
        max_tokens=512,
        request_timeout=REQUEST_TIMEOUT_S,
        max_retries=3,
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)
    return answer.strip(), docs
