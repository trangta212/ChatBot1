from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

def query_chromadb(query_text, collection_name="renthouse", persist_directory="chromaDB", top_k=5):
    # Khởi tạo embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

    # Tạo embedding cho câu truy vấn
    query_embedding = embedding_model.embed_query(query_text)

    # Kết nối tới ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_collection(collection_name)

    # Truy vấn tìm kiếm top_k kết quả tương đồng nhất
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # In kết quả
    for i in range(len(results['documents'][0])):
        print(f"--- Kết quả {i+1} ---")
        print(f"Khoảng cách similarity (cosine): {results['distances'][0][i]:.4f}")
        print("Nội dung:")
        print(results['documents'][0][i])
        print("Metadata:")
        print(results['metadatas'][0][i])
        print()

if __name__ == "__main__":
    # Ví dụ đoạn query gộp (bạn có thể thay đổi)
    query = (
        "phòng trọ ở quận 1 hồ chí minh giá dưới 5 triệu nội thất đầy đủ có điều hoà "
    )
    query_chromadb(query)
