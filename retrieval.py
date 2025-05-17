from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from underthesea import word_tokenize
import re
import unicodedata
import json

# Danh sách kiểu nhà hợp lệ
valid_house_types = ['nhatro', 'chungcumini', 'nhanguyencan', 'chungcu','canhodichvu']

# Từ điển đồng nghĩa
synonyms = {
    "phòng trọ": ["nhà trọ", "phòng cho thuê", "phòng trọ", "trọ", "nhatro"],
    "chung cư mini": ["ccmn", "chung cư mini", "căn hộ mini", "chungcumini"],
    "nhà nguyên căn": ["nhà nguyên căn", "nhà thuê", "nhà riêng", "nhanguyencan"],
    "chung cư": ["căn hộ", "chung cư", "căn hộ chung cư", "chungcu"],
    "căn hộ dịch vụ": ["can ho dich vu", "canhodichvu", "can ho dich vu", "can ho dich vu cao cap"]
}
amenities_synonyms = {
    "đầy đủ nội thất": ["đầy đủ nội thất", "full nội thất", "nội thất đầy đủ", "đầy đủ tiện nghi"],
    "điều hòa": ["điều hòa", "máy lạnh", "điều hoà", "máy điều hòa"],
    "máy giặt": ["máy giặt", "máy giặt chung", "máy giặt riêng"],
    "thang máy": ["thang máy", "có thang máy", "thang máy nội khu"],
    "hầm để xe": ["hầm để xe", "bãi để xe", "nhà xe", "chỗ để xe"],
    "gác lửng": ["gác lửng", "có gác", "gác"],
    "kệ bếp": ["kệ bếp", "bếp", "tủ bếp"],
    "không chung chủ": ["không chung chủ", "riêng chủ", "tự do chủ"],
    "giờ giấc tự do": ["giờ giấc tự do", "tự do giờ giấc", "không giới hạn giờ"]
}
# Tiền xử lý câu hỏi tiếng Việt
def preprocess_vietnamese_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = ' '.join(word_tokenize(text))
    return text

def remove_vietnamese_accents(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text

def normalize_location(text):
    if not text:
        return ""
    text = remove_vietnamese_accents(text.lower())
    text = re.sub(r'\s+', '', text)
    return text

# Hàm phân tích câu hỏi
known_districts = [
    "cầu giấy", "ba đình", "đống đa", "hoàn kiếm", "tây hồ", "hai bà trưng", "long biên",
    "thanh xuân", "hoàng mai", "hà đông", "nam từ liêm", "bắc từ liêm", "mỹ đình",
    "quận 1", "quận 3", "quận 5", "quận 7", "quận 10", "quận tân bình", "quận bình thạnh",
    "thủ đức", "gò vấp", "bình tân", "tân phú", "phú nhuận"
]

city_patterns = ["hà nội", "hanoi", "hồ chí minh", "hcm", "tphcm", "đà nẵng", "cần thơ", "hải phòng"]

def clean_location(loc_raw: str) -> str:
    if not loc_raw:
        return ""
    loc_raw = loc_raw.strip().lower()
    loc_clean = re.sub(r"^(quận|huyện|thành phố|tp\.?)\s*", "", loc_raw, flags=re.IGNORECASE)
    return loc_clean

def extract_location_from_address(address):
    if not address:
        return {}
    address_lower = address.lower()
    result = {}
    for district in known_districts:
        if district in address_lower:
            result["district"] = clean_location(district)
            break
    for city in city_patterns:
        if city in address_lower:
            result["city"] = clean_location(city)
            break
    return result

def analyze_query(query):
    query = query.lower()
    filters = {}
    text_for_vector_search = query
    
    district = None
    city = None
    for d in known_districts:
        if d in query:
            district = clean_location(d)
            # print(f"[DEBUG] Tìm thấy quận/huyện: '{d}' -> '{district}'")
            break
    for c in city_patterns:
        if c in query:
            city = clean_location(c)
            # print(f"[DEBUG] Tìm thấy thành phố: '{c}' -> '{city}'")
            break
    if district:
        filters['district'] = district
        text_for_vector_search = text_for_vector_search.replace(district, '')
    if city:
        filters['city'] = city
        text_for_vector_search = text_for_vector_search.replace(city, '')
    # print(f"[DEBUG] Vị trí cuối cùng được chọn: district='{district}', city='{city}'")
    # print(f"[DEBUG] Text sau khi xóa vị trí: '{text_for_vector_search}'")

    price_patterns = [
        (r'từ\s+(\d+(?:\.\d+)?)\s*đến\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu)', 'between'),
        (r'dưới\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu)', 'lt'),
        (r'đến\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu)', 'lte'),
        (r'trên\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu)', 'gt'),
        (r'từ\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|tribeu)', 'gte'),
    ]
    for pattern, op in price_patterns:
        match = re.search(pattern, query)
        if match:
            if op == 'between':
                filters['price_min'] = float(match.group(1)) * 1_000_000
                filters['price_max'] = float(match.group(2)) * 1_000_000
                # print(f"[DEBUG] Khoảng giá: {filters['price_min']:,}đ - {filters['price_max']:,}đ")
            else:
                price = float(match.group(1)) * 1_000_000
                if op in ['lt', 'lte']:
                    filters['price_max'] = price
                    # print(f"[DEBUG] Giá tối đa: {price:,}đ")
                else:
                    filters['price_min'] = price
                    # print(f"[DEBUG] Giá tối thiểu: {price:,}đ")
            text_for_vector_search = re.sub(pattern, '', text_for_vector_search)
            break
    
    for house_type in valid_house_types:
        if house_type in query:
            filters['type'] = house_type
            print(f"[DEBUG] Loại nhà: {house_type}")
            text_for_vector_search = text_for_vector_search.replace(house_type, '')
            break
    amenities = []

    for amenity, synonym_list in amenities_synonyms.items():
        for synonym in synonym_list:
            if synonym in query:
                amenities.append(amenity)
                print(f"[DEBUG] Tìm thấy tiện ích: '{synonym}' -> '{amenity}'")
                text_for_vector_search = text_for_vector_search.replace(synonym, '')
                break
    if amenities:
        filters['amenities'] = amenities

    text_for_vector_search = re.sub(r'\s+', ' ', text_for_vector_search.strip())
    return {
        'vector_query': text_for_vector_search,
        'filters': filters
    }

    
# Khởi tạo embedding
embedding_model = SentenceTransformerEmbeddings(
    model_name="keepitreal/vietnamese-sbert",
    model_kwargs={"device": "cpu"}
)

# Load vector DB
vector_store = Chroma(
    persist_directory="chromaDB",
    embedding_function=embedding_model,
    collection_name="renthouse"
)

def retrieve_documents(query, top_k=5):
    analysis = analyze_query(query)
    processed_query = preprocess_vietnamese_text(analysis['vector_query'])
    filters = analysis['filters']
    
    # print(f"Processed query: '{processed_query}'")
    # print(f"Filters: {filters}")
    
    # Lấy toàn bộ tài liệu từ ChromaDB
    collection = vector_store._collection.get(include=["documents", "metadatas"])
    docs = [
        {
            "document": type('Document', (), {
                'page_content': doc,
                'metadata': meta
            })(),
            "price": float(meta.get('price_per_month', float('inf'))) if meta.get('price_per_month') else float('inf')
        }
        for doc, meta in zip(collection['documents'], collection['metadatas'])
    ]
    
    # print(f"[DEBUG] Tìm thấy {len(docs)} tài liệu từ ChromaDB")
    
    filtered_docs = []
    reject_reasons = {}
    
    for i, doc_item in enumerate(docs):
        doc = doc_item["document"]
        valid = True
        doc_reasons = []
        
        # print(f"\n[DEBUG] Kiểm tra tài liệu {i+1}: {doc.metadata.get('room_name', 'N/A')}")
        
        # Kiểm tra giá (nới lỏng 20%)
        if 'price_min' in filters and 'price_per_month' in doc.metadata:
            try:
                doc_price = float(doc.metadata['price_per_month'])
                if doc_price < filters['price_min'] * 0.8:
                    reason = f"Giá {doc_price:,}đ < giá tối thiểu {filters['price_min'] * 0.8:,}đ"
                    # doc_reasons.append(reason)
                    # print(f" -> Loại: {reason}")
                    valid = False
            except (ValueError, TypeError):
                pass
        
        if 'price_max' in filters and 'price_per_month' in doc.metadata:
            try:
                doc_price = float(doc.metadata['price_per_month'])
                if doc_price > filters['price_max'] * 1.2:
                    reason = f"Giá {doc_price:,}đ > giá tối đa {filters['price_max'] * 1.2:,}đ"
                    doc_reasons.append(reason)
                    # print(f" -> Loại: {reason}")
                    valid = False
            except (ValueError, TypeError):
                pass
        
        # Kiểm tra vị trí (khớp một phần chuỗi)
        district_norm = normalize_location(doc.metadata.get('district_norm', doc.metadata.get('district', '')))
        city_norm = normalize_location(doc.metadata.get('city_norm', doc.metadata.get('city', '')))
        if (not district_norm or not city_norm) and 'address' in doc.metadata:
            extracted = extract_location_from_address(doc.metadata['address'])
            if not district_norm and 'district' in extracted:
                district_norm = normalize_location(extracted['district'])
            if not city_norm and 'city' in extracted:
                city_norm = normalize_location(extracted['city'])

        if 'district' in filters:
            filter_district_norm = normalize_location(filters['district'])
            # print(f"[DEBUG] So sánh district: '{filter_district_norm}' vs '{district_norm}'")
            if not district_norm or (filter_district_norm not in district_norm and district_norm not in filter_district_norm):
                reason = f"Quận/huyện '{filters['district']}' không khớp với '{doc.metadata.get('district', 'N/A')}'"
                doc_reasons.append(reason)
                # print(f" -> Loại: {reason}")
                valid = False

        if 'city' in filters:
            filter_city_norm = normalize_location(filters['city'])
            # print(f"[DEBUG] So sánh city: '{filter_city_norm}' vs '{city_norm}'")
            if not city_norm or (filter_city_norm not in city_norm and city_norm not in filter_city_norm):
                reason = f"Thành phố '{filters['city']}' không khớp với '{doc.metadata.get('city', 'N/A')}'"
                doc_reasons.append(reason)
                # print(f" -> Loại: {reason}")
                valid = False
        if 'type' in filters and 'type' in doc.metadata:
            filter_type = filters['type'].lower()
            doc_type = doc.metadata['type'].lower()
            
            print(f"[DEBUG] So sánh loại nhà: '{filter_type}' vs '{doc_type}'")
            
            # Kiểm tra đồng nghĩa
            matched = False
            for key, synonyms_list in synonyms.items():
                if filter_type in synonyms_list and doc_type in synonyms_list:
                    matched = True
                    break
                    
            if not matched and filter_type != doc_type:
                reason = f"Loại nhà '{filter_type}' không khớp với '{doc_type}'"
                doc_reasons.append(reason)
                print(f" -> Loại: {reason}")
                valid = False
        # Kiểm tra tiện ích
        if 'amenities' in filters:
            for amenity in filters['amenities']:
                matched = False
                # Kiểm tra full_furnishing
                if amenity == "đầy đủ nội thất" and 'full_furnishing' in doc.metadata:
                    if any(syn in doc.metadata['full_furnishing'].lower() for syn in amenities_synonyms[amenity]):
                        matched = True
                # Kiểm tra extensions
                if 'extensions' in doc.metadata:
                    if any(syn in doc.metadata['extensions'].lower() for syn in amenities_synonyms[amenity]):
                        matched = True
                # Kiểm tra chunk_content
                if any(syn in doc.page_content.lower() for syn in amenities_synonyms[amenity]):
                    matched = True
                if not matched:
                    reason = f"Không có tiện ích '{amenity}'"
                    doc_reasons.append(reason)
                    print(f" -> Loại: {reason}")
                    valid = False
                if valid:
                    print(f" -> Chấp nhận tài liệu {i+1}, chunk_id={doc.metadata.get('chunk_id', 'N/A')}")
                    filtered_docs.append(doc_item)
                else:
                    reject_reasons[i] = doc_reasons

    
    # Sắp xếp theo giá thấp nhất
    filtered_docs.sort(key=lambda x: x["price"])
    
    # Lấy top_k tài liệu
    filtered_docs = filtered_docs[:top_k]
    
    print("\n=== BÁO CÁO TÌM KIẾM ===")
    print(f"- Tổng số tài liệu tìm thấy: {len(docs)}")
    print(f"- Số tài liệu hợp lệ: {len(filtered_docs)}")
    print(f"- Số tài liệu bị loại: {len(reject_reasons)}")
    
    if reject_reasons:
        for idx, reasons in reject_reasons.items():
            doc_name = docs[idx]["document"].metadata.get('room_name', f'Tài liệu {idx+1}')
    
    if filtered_docs:
        print("\n=== CHI TIẾT TÀI LIỆU HỢP LỆ ===")
        for i, doc_item in enumerate(filtered_docs):
            doc = doc_item["document"]
            print(f"Tài liệu {i+1}:")
            print(f"  chunk_id: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"  room_name: {doc.metadata.get('room_name', 'N/A')}")
            print(f"  price_per_month: {doc.metadata.get('price_per_month', 'N/A')}")
            print(f"  address: {doc.metadata.get('address', 'N/A')}")
            print(f"  content: {doc.page_content[:200]}...\n")
            print(f"  url : {doc.metadata.get('url', 'N/A')}")
    
    if not filtered_docs:
        print("Không tìm thấy kết quả phù hợp với điều kiện lọc.")
        return []
    
    return [{"document": doc["document"], "similarity": 0.0} for doc in filtered_docs]

if __name__ == "__main__":
    queries = [
        "Tìm nhà trọ dưới 5 triệu ở có điều hoà ở hai bà trưng Hà Nội",
    ]
    for query in queries:
        try:
            print(f"\n=== Truy vấn: '{query}' ===\n")
            results = retrieve_documents(query, top_k=5)
            print(f"\nKết quả: {len(results)} tài liệu\n")
        except Exception as e:
            print(f"Lỗi khi xử lý truy vấn: {str(e)}")