from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch

from config import Paths, openai_api_key


def main():
    loader = CSVLoader(
        file_path=str(Paths.testnino1_classification_task),
        source_column="Link",
        csv_args={
            "delimiter": ",",
        },
    )
    documents = loader.load()

    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=openai_api_key, max_retries=10_000)
    db = ElasticVectorSearch.from_documents(
        documents,
        embeddings,
        elasticsearch_url="http://localhost:9200",
        index_name="elastic-index",
    )
    print(db.client.info())


if __name__ == "__main__":
    main()
