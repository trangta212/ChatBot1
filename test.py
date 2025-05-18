import chromadb
client = chromadb.PersistentClient(path="chromaDB")
collection = client.get_collection("renthouse")
results = collection.get(include=["metadatas"])
for meta in results["metadatas"]:
    print(f"chunk_id={meta['chunk_id']}, type={meta['type']}, category={meta['category']}, room_name={meta['room_name']}")
    