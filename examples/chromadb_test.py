import asyncio
import chromadb
from chromadb.api.types import IncludeEnum


async def main():
    collection_name = 'kf_chromadb_test'
    client = await chromadb.AsyncHttpClient(host='localhost', port=8888)
    try:
        await client.delete_collection(collection_name)
        print(f'deleted existing collection: {collection_name}')
    except (Exception,) as e:
        print(f'no existing collection to delete: {collection_name} {e}')

    collection = await client.create_collection(name=collection_name)
    await collection.add(
        documents=[
            "This is a document about pineapple",
            "This is a document about oranges"
        ],
        ids=["id1", "id2"]
    )

    # IncludeEnum:
    #     documents = "documents"
    #     embeddings = "embeddings"
    #     metadatas = "metadatas"
    #     distances = "distances"
    #     uris = "uris"
    #     data = "data"
    results = await collection.query(
        query_texts=["This is a query document about florida"],  # Chroma will embed this for you
        n_results=2,  # how many results to return
        include=[IncludeEnum('documents'), IncludeEnum('metadatas'), IncludeEnum('distances'), IncludeEnum('uris'), IncludeEnum('data')],
    )
    print(f'results: {results}')


asyncio.run(main())
