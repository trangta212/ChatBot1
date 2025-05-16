from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from underthesea import word_tokenize
import re
import unicodedata


# Danh sách kiểu nhà hợp lệ
valid_house_types = ['nhatro', 'chungcumini', 'nhanguyencan', 'chungcu']

# Tiền xử lý câu hỏi tiếng Việt: loại bỏ ký tự đặc biệt, tokenize
def preprocess_vietnamese_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = ' '.join(word_tokenize(text))
    return text
def remove_vietnamese_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text

def normalize_text(text):
    text = remove_vietnamese_accents(text)
    return text.lower().strip()
def normalize_location(text):
    text = remove_vietnamese_accents(text.lower())
    text = re.sub(r'\s+', '', text)  # Bỏ hết khoảng trắng
    return text
# Hàm phân tích câu hỏi tách điều kiện lọc (giá, diện tích, vị trí, kiểu nhà)
def analyze_query(query):
    query = query.lower()
    filters = {}
    text_for_vector_search = query
    # Xử lý vị trí
    known_districts = [
        "cầu giấy", "ba đình", "đống đa", "hoàn kiếm", "tây hồ", "hai bà trưng", "long biên",
        "thanh xuân", "hoàng mai", "hà đông",
        "quận 1", "quận 3", "quận 5", "quận 7", "quận 10", "quận tân bình", "quận bình thạnh",
        "thủ đức", "gò vấp"
    ]
    city_patterns = ["hà nội", "hanoi", "hồ chí minh", "hcm", "đà nẵng", "cần thơ", "hải phòng"]

    def clean_location(loc_raw: str):
        loc_raw = loc_raw.strip()
        # Loại bỏ tiền tố phổ biến với ignore case
        loc_clean = re.sub(r"^(quận|huyện|thành phố|tp\.?|tp)\s*", "", loc_raw, flags=re.I)
        return loc_clean

    location = None
    # Tìm quận/huyện trong câu hỏi
    for district in known_districts:
        if district in query:
            location = clean_location(district)
            break
    # Nếu không tìm quận/huyện thì tìm city
    if not location:
        for city in city_patterns:
            if city in query:
                location = city
                break

    if location:
        filters['location'] = location
        # Xóa location trong câu vector search
        text_for_vector_search = text_for_vector_search.replace(location, '')

    # Xử lý giá
    price_patterns = [
        (r'từ\s+(\d+(?:\.\d+)?)\s*đến\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu)', 'between'),
        (r'dưới\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu)', 'lt'),
        (r'đến\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu)', 'lte'),
        (r'trên\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu)', 'gt'),
        (r'từ\s+(\d+(?:\.\d+)?)\s*(?:triệu|tr|trieu)', 'gte'),
    ]

    for pattern, op in price_patterns:
        match = re.search(pattern, query)
        if match:
            if op == 'between':
                filters['price_min'] = float(match.group(1)) * 1_000_000
                filters['price_max'] = float(match.group(2)) * 1_000_000
            else:
                price = float(match.group(1)) * 1_000_000
                if op in ['lt', 'lte']:
                    filters['price_max'] = price
                else:
                    filters['price_min'] = price
            text_for_vector_search = re.sub(pattern, '', text_for_vector_search)
            break

    # Xử lý diện tích
    area_patterns = [
        (r'từ\s+(\d+(?:\.\d+)?)\s*đến\s+(\d+(?:\.\d+)?)\s*(?:m2|mét vuông|met vuong)', 'between'),
        (r'dưới\s+(\d+(?:\.\d+)?)\s*(?:m2|mét vuông|met vuong)', 'lt'),
        (r'đến\s+(\d+(?:\.\d+)?)\s*(?:m2|mét vuông|met vuong)', 'lte'),
        (r'trên\s+(\d+(?:\.\d+)?)\s*(?:m2|mét vuông|met vuong)', 'gt'),
        (r'từ\s+(\d+(?:\.\d+)?)\s*(?:m2|mét vuông|met vuong)', 'gte'),
    ]

    for pattern, op in area_patterns:
        match = re.search(pattern, query)
        if match:
            if op == 'between':
                filters['area_min'] = float(match.group(1))
                filters['area_max'] = float(match.group(2))
            else:
                area = float(match.group(1))
                if op in ['lt', 'lte']:
                    filters['area_max'] = area
                else:
                    filters['area_min'] = area
            text_for_vector_search = re.sub(pattern, '', text_for_vector_search)
            break

    # Xử lý kiểu nhà
    for house_type in valid_house_types:
        if house_type in query:
            filters['type'] = house_type
            text_for_vector_search = text_for_vector_search.replace(house_type, '')
            break

    # Làm sạch
    text_for_vector_search = re.sub(r'\s+', ' ', text_for_vector_search.strip())
    return {
        'vector_query': text_for_vector_search,
        'filters': filters
    }

# Khởi tạo embedding giống khi lưu ChromaDB
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
def remove_vietnamese_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text

def normalize_text(text):
    text = remove_vietnamese_accents(text)
    return text.lower().strip()

def retrieve_documents(query, top_k=10):
    analysis = analyze_query(query)
    processed_query = preprocess_vietnamese_text(analysis['vector_query'])
    filters = analysis['filters']
    
    print(f"Vector search query: '{processed_query}'")
    print(f"Filters: {filters}")
    
    docs = vector_store.similarity_search(processed_query, k=top_k*3)
    
    filtered_docs = []
    for i, doc in enumerate(docs):
        valid = True
        print(f"Metadata: {doc.metadata}")
        
        # Kiểm tra giá
        if 'price_min' in filters and 'price_per_month' in doc.metadata:
            if float(doc.metadata['price_per_month']) < filters['price_min']:
                print(f" -> Loại do price_per_month {doc.metadata['price_per_month']} < price_min {filters['price_min']}")
                valid = False
        
        if 'price_max' in filters and 'price_per_month' in doc.metadata:
            if float(doc.metadata['price_per_month']) > filters['price_max']:
                print(f" -> Loại do price_per_month {doc.metadata['price_per_month']} > price_max {filters['price_max']}")
                valid = False
        
        # Kiểm tra diện tích
        if 'area_min' in filters and 'area' in doc.metadata:
            if float(doc.metadata['area']) < filters['area_min']:
                print(f" -> Loại do area {doc.metadata['area']} < area_min {filters['area_min']}")
                valid = False
        
        if 'area_max' in filters and 'area' in doc.metadata:
            if float(doc.metadata['area']) > filters['area_max']:
                print(f" -> Loại do area {doc.metadata['area']} > area_max {filters['area_max']}")
                valid = False
        
        # Kiểm tra vị trí
            # Kiểm tra vị trí (địa chỉ metadata)
        if 'location' in filters:
            filter_loc_norm = normalize_location(filters['location'])
            district_norm = normalize_location(doc.metadata.get('district_norm', ''))
            city_norm = normalize_location(doc.metadata.get('city_norm', ''))
        if filter_loc_norm not in district_norm and filter_loc_norm not in city_norm:
            print(f" -> Loại do location '{filter_loc_norm}' không trùng district '{district_norm}' hay city '{city_norm}'")
            valid = False

        # Kiểm tra kiểu nhà
        if 'type' in filters and 'type' in doc.metadata:
            if filters['type'] != doc.metadata['type'].lower():
                print(f" -> Loại do type '{doc.metadata['type'].lower()}' != filter '{filters['type']}'")
                valid = False
        
        if valid:
            print(" -> Chấp nhận document này")
            filtered_docs.append(doc)
            if len(filtered_docs) >= top_k:
                break
        else:
            print(" -> Document bị loại")
    
    if not filtered_docs and docs:
        print("Không tìm thấy kết quả phù hợp với điều kiện lọc. Trả về kết quả tìm kiếm vector.")
        return docs[:top_k]
    
    return filtered_docs

# Ví dụ test
if __name__ == "__main__":
    queries = [
        "Tìm phòng trọ dưới 10 triệu ở quận hai bà trưng Hà Nội",
    ]

    for query in queries:
        print(f"\n=== Truy vấn: '{query}' ===\n")
        results = retrieve_documents(query)

        print(f"\nKết quả: {len(results)} tài liệu\n")
        for i, doc in enumerate(results):
            print(f"Tài liệu {i+1}: {doc.page_content}\nMetadata: {doc.metadata}\n")
