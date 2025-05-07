import asyncio

import chromadb


async def main():
    collection_name = 'kf_chromadb_test'
    client = await chromadb.AsyncHttpClient(host='localhost', port=8888)
    try:
        await client.delete_collection(collection_name)
        print(f'deleted existing collection: {collection_name}')
    except (Exception,) as e:
        print(f'no existing collection to delete: {collection_name}: {e.__class__.__name__}: {e}')

    collection = await client.create_collection(name=collection_name,
                                                metadata={
                                                    "hnsw:space": "cosine",
                                                    "hnsw:construction_ef": 600,
                                                    "hnsw:search_ef": 1000,
                                                    "hnsw:M": 60
                                                },
                                                )
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
    query = "This is a query document about florida"
    print(f'running query: {query}...')
    results = await collection.query(
        query_texts=[query],  # Chroma will embed this for you
        n_results=2,  # how many results to return
        include=['documents', 'metadatas', 'distances', 'uris', 'data'],
    )
    print(f'results: {results}')


asyncio.run(main())
