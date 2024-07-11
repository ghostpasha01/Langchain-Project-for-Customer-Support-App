from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class HelpCenterAgent:
    FREE_SUB_DOC_PATH = "assets/free"
    PREMIUM_SUB_DOC_PATH = "assets/paid"

    def __init__(self):
        self._free_sub_db = self._create_index(directory=self.FREE_SUB_DOC_PATH)
        self._paid_sub_db = self._create_index(directory=self.PREMIUM_SUB_DOC_PATH)

        self._qa_chain = self._create_qa_chain()

    def _create_index(self, directory: str):
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        persist_directory = f"chroma_db/{directory}"
        docs = self.split_docs(self.load_docs(directory))

        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persist_directory
        )

        vectordb.persist()

        return vectordb

    def free_sub_retriever(self):
        return self._free_sub_db.as_retriever()

    def paid_sub_retriever(self):
        return self._paid_sub_db.as_retriever()

    def _run_query(self, vectordb: VectorStore, query: str):
        matching_docs_score = vectordb.similarity_search_with_score(query)

        matching_docs = [doc for doc, score in matching_docs_score]
        answer = self._qa_chain.run(input_documents=matching_docs, question=query)

        # Prepare the sources
        sources = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in matching_docs_score
        ]

        return {"answer": answer, "sources": sources}

    @classmethod
    def _create_qa_chain(cls):
        model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model_name=model_name)

        chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

        return chain

    @classmethod
    def load_docs(cls, directory: str):
        """
        Load documents from the given directory.
        """
        loader = DirectoryLoader(directory)
        documents = loader.load()

        return documents

    @classmethod
    def split_docs(cls, documents, chunk_size=2000, chunk_overlap=500):
        """
        Split the documents into chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = text_splitter.split_documents(documents)

        return docs
