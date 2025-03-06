from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# can generate sample documents
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# can load / integration w hundreds of common sources
file_path = "/Users/sophi/langchain-learning/content/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# pdfloader loads one document object per PDF page
print(len(docs))

# string content of a page and metadata containing file name and page number
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)


# info retrieval and downstream q&a answering -- a given page may be too coarse
# the goal is to retrieve document objects that anaswer an input query
# further splitting PDF ensures meanings of relevant portions of the documents are not "washed out"
# text splitters partitions based on parameters (i.e characters) -> chunks documents into 1000 chs with 200 chs of overlap

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)
print("len 2: ", len(all_splits))