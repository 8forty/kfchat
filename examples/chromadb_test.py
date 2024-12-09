import asyncio
import chromadb


async def main():
    client = await chromadb.AsyncHttpClient(host='localhost', port=8888)
    collection = await client.create_collection(name="kf1_collection")
    await collection.add(
        documents=[
            "This is a document about pineapple",
            "This is a document about oranges"
        ],
        ids=["id1", "id2"]
    )

    results = await collection.query(
        query_texts=["This is a query document about hawaii"],  # Chroma will embed this for you
        n_results=2  # how many results to return
    )
    print(f'results: {results}')


asyncio.run(main())
