from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import re
from datetime import datetime

# load research paper
loader = PyPDFLoader("research_paper.pdf")
documents = loader.load()

# document level metadata
# this metadata will be attached to all chunks to provide context
doc_metadata = {
    "title": "Neural Networks for Information Retrieval",
    "authors": "Smith J., Johnson K.",
    "publication_date": "2023-05-15",
    "source": "research_paper.pdf",
    "processed_date": datetime.now().isoformat()
}

# defines section pattern to identify potential headers
# looks for numbered sections
section_pattern = re.compile(r'^(\d+\.?\d*)\s+([A-Z][\w\s]+)')

#data preprocessing
enhanced_docs = []
current_section = "Introduction"
current_section_num = "1"

for doc in documents: #iterates through each page/document in the pdf
    page_content = doc.page_content
    page_num = doc.metadata.get("page")
    
    # see if it matches the header pattern we defined above
    section_match = section_pattern.search(page_content[:100])
    if section_match: #tracks current section name and number
        current_section_num = section_match.group(1)
        current_section = section_match.group(2).strip()
    
    # Enhance with rich metadata
    enhanced_metadata = {
        **doc_metadata, # the document level data we defined above
        **doc.metadata,
        "page": page_num,
        "section_number": current_section_num,
        "section_name": current_section,
        # tries to infer what kind of data below will be
        "contains_code": bool(re.search(r'```[\s\S]+```', page_content)),
        "contains_tables": "Table" in page_content or "Figure" in page_content,
        "contains_equations": bool(re.search(r'\$\$[\s\S]+\$\$', page_content))
    }
    
    # raw documents -> enriched documents with metadata woohoo
    enhanced_docs.append({
        "page_content": page_content,
        "metadata": enhanced_metadata
    })

# text splitter 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]  #custom separators that respect document structure (i.e section headers, paragraphs)
)

# chunking but with enriched metadata that we have above
'''
Which document it came from
Which page and section it belongs to
What content types it contains
Where it sits in the original document sequence
'''
all_splits = text_splitter.split_documents(enhanced_docs)
print(f"Created {len(all_splits)} chunks with rich metadata")

# convert this into embedding (numberical rep of its meaning)
# stored in chroma vector db
embeddings = OpenAIEmbeddings()
vector_db = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name="research_papers",
    persist_directory="./chroma_db"
)

# demo of retrieval filtered by metadata
query = "What are the performance metrics for neural networks in IR tasks?"
results = vector_db.similarity_search(
    query=query,
    k=3,
    filter={"section_name": "Evaluation", "contains_tables": True}
)

# shows!
for i, doc in enumerate(results):
    print(f"\nResult {i+1} from {doc.metadata['section_name']} (page {doc.metadata['page']}):")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")