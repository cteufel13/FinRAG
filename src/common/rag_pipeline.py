from langchain.document_loaders import PDFMinerLoader, UnstructuredPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA
from typing import List
import json

from src.utils.utils import parse_html_table, convert_table_row_to_text


class RagPipeline:

    def __init__(
        self,
    ):

        self.pdf_path = None
        self.embedding_model = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.qa_chain = None

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF file and return a Document object."""
        loader = UnstructuredPDFLoader(
            pdf_path,
            mode="elements",
            infer_table_structure=True,
        )
        document = loader.load()

        return document

    def chunk_text(
        self, documents: List[Document], chunk_size=1000, chunk_overlap=200
    ) -> list[Document]:
        """Chunk the text into smaller pieces."""

        Tables = []
        Text = []

        # Split the documents into tables and text
        for element in documents:

            # Check if the element is a table or text
            if element.metadata["category"] == "Table":

                table = parse_html_table(element.metadata["text_as_html"])

                for row in table:
                    content = json.dumps(row)
                    content = convert_table_row_to_text(content)

                    metadata = element.metadata.copy()
                    metadata["category"] = "TableRow"
                    Tables.append(Document(page_content=content, metadata=metadata))

            elif element.metadata["category"] in [
                "NarrativeText",
            ]:
                Text.append(element)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        )
        chunks = text_splitter.split_documents(Text + Tables)
        return chunks

    def create_vectordb(
        self,
        chunks: list[Document],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        search_kwargs: dict = {},
    ) -> FAISS:
        """Create a vector database from the chunks."""
        self.embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model)
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        return self.vectorstore

    def load_llm(
        self,
        llm_model: str = "google/flan-t5-base",
        pipeline_type="text2text-generation",
    ) -> HuggingFacePipeline:
        """Load the LLM model."""
        llm_pipeline = pipeline(
            task=pipeline_type,
            model=llm_model,
            max_length=512,
        )
        self.llm = HuggingFacePipeline(pipeline=llm_pipeline)

        return self.llm

    def create_qa_chain(
        self,
        chain_type="stuff",
        chain_type_kwargs: dict = {},
        return_source_documents=True,
    ) -> RetrievalQA:
        """Create a QA chain."""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=return_source_documents,
        )

        return self.qa_chain

    def run(self, query: str) -> str:
        """Run the QA chain with a query."""
        if self.qa_chain is None:
            raise ValueError("QA chain is not created. Please create it first.")
        query_result = self.qa_chain(query)
        answer = query_result["result"]
        stuff = query_result["source_documents"]

        return answer, stuff
