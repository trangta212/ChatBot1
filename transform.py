import json
import re
import unicodedata
from sentence_transformers import SentenceTransformer  # type: ignore
import chromadb  # type: ignore
from chromadb.config import Settings  # type: ignore


def clean_extensions(extensions_raw):
    if isinstance(extensions_raw, list):
        return ", ".join([e.strip() for e in extensions_raw if isinstance(e, str) and e.strip()])
    elif isinstance(extensions_raw, str):
        return ", ".join([e.strip() for e in extensions_raw.split(",") if e.strip()])
    return ""


def remove_vietnamese_accents(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def normalize_location_text(text):
    # Loại bỏ dấu, viết thường, bỏ khoảng trắng, nối liền
    if not text:
        return ""
    text = remove_vietnamese_accents(text)
    text = text.lower()
    text = re.sub(r"\s+", "", text)  # bỏ mọi khoảng trắng
    return text


def parse_address(address):
    """
    Tách địa chỉ thành street, ward, district, city
    Xử lý kỹ hơn, tách dấu phẩy, làm thường, không dấu, nối liền
    """
    street = ward = district = city = ""
    if not address:
        return street, ward, district, city, "", "", ""

    parts = [p.strip() for p in address.split(",")]

    if len(parts) >= 1:
        city_raw = parts[-1]
    else:
        city_raw = ""

    if len(parts) >= 2:
        district_raw = parts[-2]
    else:
        district_raw = ""

    if len(parts) >= 3:
        ward_raw = parts[-3]
    else:
        ward_raw = ""

    if len(parts) >= 4:
        street_raw = ", ".join(parts[:-3])
    else:
        street_raw = ""

    # Loại bỏ tiền tố phổ biến ở district và ward
    district_raw = re.sub(r"^(quận|huyện|thành phố|tp\.?)\s*", "", district_raw, flags=re.I)
    ward_raw = re.sub(r"^(phường|xã|thị trấn)\s*", "", ward_raw, flags=re.I)

    # Chuẩn hóa cho việc so sánh, lưu metadata chuẩn hóa (đã remove dấu, viết liền)
    district_norm = normalize_location_text(district_raw)
    ward_norm = normalize_location_text(ward_raw)
    city_norm = normalize_location_text(city_raw)

    return street_raw, ward_raw, district_raw, city_raw, ward_norm, district_norm, city_norm


def save_to_chromadb(json_path, collection_name="renthouse"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = SentenceTransformer("keepitreal/vietnamese-sbert")

    texts = []
    ids = []
    metadatas = []

    for i, item in enumerate(data):
        content_parts = []

        if "room_name" in item:
            content_parts.append(item["room_name"])
        if "address" in item:
            content_parts.append(item["address"])
        if "chunk_content" in item:
            content_parts.append(item["chunk_content"])

        extensions_clean = clean_extensions(item.get("extensions", ""))
        if extensions_clean:
            content_parts.append("Tiện ích: " + extensions_clean)

        if "full_furnishing" in item:
            content_parts.append("Nội thất: " + item["full_furnishing"])
        if "url" in item:
            content_parts.append(item["url"])

        content = "\n".join(content_parts)

        texts.append(content)
        ids.append(f"doc_{i}")

        street, ward_raw, district_raw, city_raw, ward_norm, district_norm, city_norm = parse_address(item.get("address", ""))

        metadata = {
            "room_name": item.get("room_name", ""),
            "url": item.get("url", ""),
            "address": item.get("address", ""),
            "street": street,
            "ward": ward_raw,
            "district": district_raw,
            "city": city_raw,
            "ward_norm": ward_norm,
            "district_norm": district_norm,
            "city_norm": city_norm,
            "price_per_month": item.get("price_per_month", 0),
            "area": item.get("area", 0),
            "extensions": extensions_clean,
            "full_furnishing": item.get("full_furnishing", ""),
            "start_date": item.get("start_date", ""),
            "expire": item.get("expire", ""),
            "latitude": item.get("latitude", 0.0),
            "longitude": item.get("longitude", 0.0),
            "type": item.get("type", ""),
            "chunk_id": item.get("chunk_id", 0),
        }
        metadatas.append(metadata)

    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    chroma_client = chromadb.PersistentClient(path="chromaDB")
    collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    batch_size = 100

    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )

    print(f"✅ Đã lưu {len(texts)} chunks vào ChromaDB trong collection '{collection_name}'")


if __name__ == "__main__":
    save_to_chromadb("output_chunks.json")
