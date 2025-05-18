from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from underthesea import word_tokenize
import re
import unicodedata
import math
import requests
from geopy.geocoders import Nominatim
from functools import lru_cache

# Danh sách kiểu nhà hợp lệ
valid_house_types = ['nhatro', 'chungcumini', 'nhanguyencan', 'chungcu', 'canhodichvu']

# Từ điển đồng nghĩa cho kiểu nhà
synonyms = {
    "phòng trọ": ["nhà trọ", "phòng cho thuê", "phòng trọ", "trọ", "nhatro"],
    "chung cư mini": ["ccmn", "chung cư mini", "căn hộ mini", "chungcumini"],
    "nhà nguyên căn": ["nhà nguyên căn", "nhà thuê", "nhà riêng", "nhanguyencan"],
    "chung cư": ["căn hộ", "chung cư", "căn hộ chung cư", "chungcu"],
    "căn hộ dịch vụ": ["can ho dich vu", "canhodichvu", "can ho dich vu", "can ho dich vu cao cap"]
}

# Từ điển đồng nghĩa cho tiện ích
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

# Từ khóa cho truy vấn quy trình
process_keywords = {
    "hướng dẫn đăng bài": [
        "hướng dẫn đăng bài", "cách đăng bài", "đăng bài như thế nào", 
        "hướng dẫn đăng phòng", "làm sao để đăng phòng", "đăng tin phòng", 
        "hướng dẫn đăng tin", "cách đăng tin phòng","cách đăng tin", "đăng tin phòng như thế nào",
        "đăng tin ntn"
    ],
    "phương thức thanh toán": [
        "quy trình thanh toán", "cách thanh toán", "thanh toán như thế nào", 
        "trả tiền ra sao", "quy trình trả tiền", "thanh toán tiền phòng"
    ],
    "điều khoản hợp đồng": [
        "điều khoản hợp đồng", "hợp đồng thuê", "điều khoản thuê", 
        "quy định hợp đồng", "điều khoản thuê phòng", "hợp đồng thuê nhà"
    ],
    "hướng dẫn đặt phòng": [
        "hướng dẫn đặt phòng", "cách đặt phòng", "đặt phòng như thế nào",
        "hướng dẫn thuê phòng", "làm sao để thuê phòng", "đặt phòng như thế nào",
        "hướng dẫn thuê phòng", "cách thuê phòng"
    ],
    "Quản lý bài đăng": [
        "quản lý bài đăng", "cách quản lý bài đăng", "quản lý phòng",
        "quản lý tin đăng", "quản lý bài viết", "quản lý bài viết như thế nào",
        "quản lý bài viết", "quản lý bài viết như thế nào"
    ],
    "chính sách hoàn tiền": [
        "chính sách hoàn tiền", "cách hoàn tiền", "hoàn tiền như thế nào",
        "chính sách hoàn tiền", "hoàn tiền", "hoàn tiền như thế nào"
    ],
    "lưu ý khi đăng bài": [
        "lưu ý khi đăng bài", 
        "lưu ý khi đăng phòng", "lưu ý khi đăng tin", 
        "lưu ý", "những lưu ý khi đăng"
    ],
}

# Danh sách quận/huyện được biết
known_districts = [
    # Hà Nội
    "ba đình", "hoàn kiếm", "tây hồ", "long biên", "cầu giấy", "đống đa", "hai bà trưng", 
    "hoàng mai", "thanh xuân", "hà đông", "bắc từ liêm", "nam từ liêm", "sơn tây",
    "ba vì", "chương mỹ", "đan phượng", "đông anh", "gia lâm", "hoài đức", "mê linh", 
    "mỹ đức", "phú xuyên", "phúc thọ", "quốc oai", "sóc sơn", "thạch thất", "thanh oai", 
    "thanh trì", "thường tín", "ứng hoà", "mỹ đình",
    
    # Hồ Chí Minh 
    "quận 1", "quận 2", "quận 3", "quận 4", "quận 5", "quận 6", "quận 7", "quận 8",
    "quận 9", "quận 10", "quận 11", "quận 12", "bình tân", "bình thạnh", "gò vấp", 
    "phú nhuận", "tân bình", "tân phú", "thủ đức", "bình chánh", "cần giờ", "củ chi",
    "hóc môn", "nhà bè", "thủ đức", "quận tân bình", "quận bình thạnh",
    
    # Đà Nẵng
    "hải châu", "thanh khê", "sơn trà", "ngũ hành sơn", "liên chiểu", "cẩm lệ", "hòa vang",
    
    # Hải Phòng
    "hồng bàng", "lê chân", "ngô quyền", "kiến an", "hải an", "đồ sơn", "dương kinh",
    
    # Cần Thơ
    "ninh kiều", "bình thủy", "cái răng", "ô môn", "thốt nốt"
]

# Danh sách thành phố được biết
city_patterns = [
    # Hà Nội và biến thể
    "hà nội", "hanoi", "ha noi", "thủ đô", "thu do", "hn",
    
    # Hồ Chí Minh và biến thể
    "hồ chí minh", "ho chi minh", "tp hcm", "tphcm", "hcm", "sài gòn", "sai gon", "tp.hcm", "sg", 
    "thành phố hồ chí minh", "tp. hồ chí minh", "thanh pho ho chi minh",
    
    # Đà Nẵng và biến thể
    "đà nẵng", "da nang", "tp đà nẵng", "tp da nang", "tp. đà nẵng", "đn",
    
    # Cần Thơ và biến thể
    "cần thơ", "can tho", "tp cần thơ", "tp can tho", "tp. cần thơ", "ct",
    
    # Hải Phòng và biến thể
    "hải phòng", "hai phong", "tp hải phòng", "tp hai phong", "tp. hải phòng", "hp",
    
    # Các thành phố lớn khác
    "huế", "hue", "thừa thiên huế", "thua thien hue",
    "nha trang", "khánh hòa", "khanh hoa",
    "đà lạt", "da lat", "lâm đồng", "lam dong",
    "vũng tàu", "vung tau", "bà rịa vũng tàu", "ba ria vung tau",
    "biên hòa", "bien hoa", "đồng nai", "dong nai",
    "hải dương", "hai duong",
    "hà long", "ha long", "quảng ninh", "quang ninh",
    "thái nguyên", "thai nguyen",
    "vinh", "nghệ an", "nghe an",
    "quy nhơn", "quy nhon", "bình định", "binh dinh",
    "long xuyên", "long xuyen", "an giang",
    "buôn ma thuột", "buon ma thuot", "đắk lắk", "dak lak", "daklak",
    "rạch giá", "rach gia", "kiên giang", "kien giang",
    "mỹ tho", "my tho", "tiền giang", "tien giang",
    "nam định", "nam dinh",
    "phan thiết", "phan thiet", "bình thuận", "binh thuan",
    "pleiku", "gia lai",
    "tây ninh", "tay ninh",
    "thái bình", "thai binh",
    "việt trì", "viet tri", "phú thọ", "phu tho"
]

def haversine_distance(lat1, lon1, lat2, lon2):
    """Tính khoảng cách giữa hai điểm dựa vào kinh độ, vĩ độ (đơn vị: km)"""
    # Chuyển đổi độ sang radian
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Công thức haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Bán kính Trái Đất tính bằng km
    return c * r

@lru_cache(maxsize=128)
def geocode_address(address, city=None):
    """Chuyển địa chỉ thành tọa độ (kinh độ, vĩ độ)"""
    try:
        # Thêm tên thành phố nếu có để tăng độ chính xác
        if city:
            full_address = f"{address}, {city}, Việt Nam"
        else:
            full_address = f"{address}, Việt Nam"
            
        geolocator = Nominatim(user_agent="rental_assistant")
        location = geolocator.geocode(full_address)
        
        if location:
            return {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "address": location.address
            }
        else:
            # Backup với OpenStreetMap API nếu Nominatim không trả kết quả
            url = f"https://nominatim.openstreetmap.org/search?q={full_address}&format=json&limit=1"
            response = requests.get(url)
            if response.status_code == 200 and response.json():
                result = response.json()[0]
                return {
                    "latitude": float(result['lat']),
                    "longitude": float(result['lon']),
                    "address": result.get('display_name', full_address)
                }
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
    
    return None

def extract_radius_from_query(query):
    """Trích xuất bán kính từ câu truy vấn"""
    radius_patterns = [
        (r'trong\s+(?:vòng|bán\s+kính)\s+(\d+(?:\.\d+)?)\s*(?:km|kilomet|kilometer)', 1),
        (r'(?:bán\s+kính|khoảng\s+cách)\s+(\d+(?:\.\d+)?)\s*(?:km|kilomet|kilometer)', 1),
        (r'cách\s+(?:khoảng|tầm)\s+(\d+(?:\.\d+)?)\s*(?:km|kilomet|kilometer)', 1),
        (r'(\d+(?:\.\d+)?)\s*(?:km|kilomet|kilometer)', 1)  # Pattern đơn giản hơn, ưu tiên thấp
    ]
    
    for pattern, group in radius_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return float(match.group(group))
    
    return 3.0  # Bán kính mặc định 3km

def extract_address_from_query(query):
    """Trích xuất địa chỉ từ truy vấn người dùng"""
    address_patterns = [
        (r'gần\s+(.+?)(?:\s+trong\s+vòng|\s+trong\s+bán\s+kính|\s+với|\s+có|\s+giá|\s+dưới|\s+từ|\.$|$)', 1),
        (r'gần\s+địa\s+chỉ\s+(.+?)(?:\s+trong\s+vòng|\s+trong\s+bán\s+kính|\s+với|\s+có|\s+giá|\s+dưới|\s+từ|\.$|$)', 1),
        (r'quanh\s+(.+?)(?:\s+trong\s+vòng|\s+trong\s+bán\s+kính|\s+với|\s+có|\s+giá|\s+dưới|\s+từ|\.$|$)', 1),
        (r'khu\s+vực\s+(.+?)(?:\s+trong\s+vòng|\s+trong\s+bán\s+kính|\s+với|\s+có|\s+giá|\s+dưới|\s+từ|\.$|$)', 1),
        (r'tại\s+(.+?)(?:\s+trong\s+vòng|\s+trong\s+bán\s+kính|\s+với|\s+có|\s+giá|\s+dưới|\s+từ|\.$|$)', 1),
        (r'(?:địa chỉ|địa điểm|vị trí)\s+(.+?)(?:\s+với|\s+có|\s+giá|\s+dưới|\s+từ|\.$|$)', 1)
    ]
    
    for pattern, group in address_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            address = match.group(group).strip()
            # Loại bỏ các từ khóa không phải địa chỉ
            noise_words = ["giá rẻ", "phòng trọ", "căn hộ", "giá tốt", "gần đây"]
            for word in noise_words:
                address = address.replace(word, "").strip()
            return address
    
    return None

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
    """Chuẩn hóa vị trí để so sánh, đặc biệt xử lý chính xác cho quận số"""
    if not text:
        return ""
    
    text = text.lower().strip()
    
    # Xử lý đặc biệt cho quận số - CẢI TIẾN để bắt đúng toàn bộ số
    quan_patterns = [
        (r"quận\s+(\d+)", "quan{0}"),  # quận 3 -> quan3, quận 10 -> quan10
        (r"q\.?\s*(\d+)", "quan{0}"),  # q3, q.3, q10, q.10
        (r"quan\s*(\d+)", "quan{0}"),  # quan 3 -> quan3
    ]
    
    for pattern, template in quan_patterns:
        quan_match = re.search(pattern, text, re.IGNORECASE)
        if quan_match:
            district_number = quan_match.group(1)
            result = template.format(district_number)
            # Debug để xác nhận chuỗi số đúng
            return result
    
    # Xử lý các trường hợp không phải quận số
    result = remove_vietnamese_accents(text)
    result = re.sub(r'\s+', '', result)
    return result
def contains_district_number(text, number):
    # Tìm đúng "quận 10", không khớp "quận 1"
    pattern = r"\bquận\s*0*{}\b".format(number)
    return re.search(pattern, text.lower()) is not None
def extract_all_district_numbers(text):
    # Trả về list các số quận xuất hiện trong text
    return re.findall(r"\bquận\s*0*(\d{1,2})\b", text.lower())

def clean_location(loc_raw: str) -> str:
    """
    Loại bỏ tiền tố như 'quận', 'huyện', 'thành phố', 'tp' khỏi chuỗi vị trí.
    Xử lý đặc biệt cho quận số để tránh nhầm lẫn.
    """
    if not loc_raw:
        return ""
    
    loc_raw = loc_raw.strip().lower()
    
    # Xử lý đặc biệt cho quận số
    quan_match = re.search(r"quận\s+(\d+)", loc_raw, re.IGNORECASE)
    if quan_match:
        return f"quận {quan_match.group(1)}"  # Giữ nguyên "quận X" để tránh nhầm lẫn
    
    # Xử lý các trường hợp khác
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
    
    # Kiểm tra truy vấn về quy trình
    for category, keywords in process_keywords.items():
        for keyword in keywords:
            if keyword in query:
                filters['process_category'] = category
                text_for_vector_search = text_for_vector_search.replace(keyword, '')
                break
        if 'process_category' in filters:
            break

    # Nếu không phải truy vấn quy trình, xử lý các bộ lọc phòng
    if 'process_category' not in filters:
        # Trích xuất kinh độ, vĩ độ, bán kính
        lat_lon_pattern = r"(?:kinh độ|longitude|lon)\s*[:=]?\s*([-]?\d+\.?\d*)\s*(?:vĩ độ|latitude|lat)\s*[:=]?\s*([-]?\d+\.?\d*)"
        
        # Trích xuất bán kính tìm kiếm
        radius = extract_radius_from_query(query)
        
        lat_lon_match = re.search(lat_lon_pattern, query)
        
        if lat_lon_match:
            filters['location'] = {
                'latitude': float(lat_lon_match.group(2)),
                'longitude': float(lat_lon_match.group(1)),
                'radius': radius
            }
            text_for_vector_search = re.sub(lat_lon_pattern, '', text_for_vector_search)

        # Trích xuất quận/huyện, thành phố
        district = None
        city = None
        # for d in known_districts:
        #     if d in query:
        #         district = clean_location(d)
        #         print(f"[DEBUG] Tìm thấy quận/huyện: '{d}' -> '{district}'")
        #         break
         # Ưu tiên xử lý quận số trước để đảm bảo bắt đúng
        district_number_pattern = r"quận\s+(\d+)"
        district_match = re.search(district_number_pattern, query.lower())
        
        if district_match:
            district_number = district_match.group(1)
            district = f"quận {district_number}"  # Giữ nguyên định dạng "quận X"
            
            # Thử nghiệm normalize để xác nhận kết quả
            district_norm = normalize_location(district)
        else:
            # Thử các dạng khác của quận số như q10, q.10
            q_number_pattern = r"q\.?\s*(\d+)"
            q_match = re.search(q_number_pattern, query.lower())
            
            if q_match:
                district_number = q_match.group(1)
                district = f"quận {district_number}"
            else:
                # Tìm quận/huyện thông thường
                for d in known_districts:
                    if d in query.lower():
                        district = clean_location(d)
                        break
        
        for c in city_patterns:
            if c in query:
                city = clean_location(c)
                break
        if district:
            filters['district'] = district
            text_for_vector_search = text_for_vector_search.replace(district, '')
        if city:
            filters['city'] = city
            text_for_vector_search = text_for_vector_search.replace(city, '')

        # Xử lý địa chỉ người dùng
        user_address = extract_address_from_query(query)
        if user_address:
            
            # Chuyển địa chỉ thành tọa độ
            coords = geocode_address(user_address, city)
            if coords:
                filters['location'] = {
                    'latitude': coords['latitude'], 
                    'longitude': coords['longitude'],
                    'radius': radius,
                    'address': coords['address']
                }

                
                # Đánh dấu từ khóa địa chỉ để loại bỏ khỏi vector search
                text_for_vector_search = text_for_vector_search.replace(user_address, '')

        # Trích xuất giá
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
        
        # Trích xuất loại nhà
        for house_type, synonym_list in synonyms.items():
            for synonym in synonym_list:
                if synonym in query:
                    # Ánh xạ từ khóa thành giá trị trong valid_house_types
                    for valid_type in valid_house_types:
                        if valid_type in synonym_list:
                            filters['type'] = valid_type
                            text_for_vector_search = text_for_vector_search.replace(synonym, '')
                            break
                    break
            if 'type' in filters:
                break

        # Trích xuất tiện ích
        amenities = []
        for amenity, synonym_list in amenities_synonyms.items():
            for synonym in synonym_list:
                if synonym in query:
                    amenities.append(amenity)
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

def retrieve_documents(query, top_k=3):
    analysis = analyze_query(query)
    processed_query = preprocess_vietnamese_text(analysis['vector_query'])
    filters = analysis['filters']
    

    # Nếu là truy vấn quy trình, thử tìm kiếm ngữ nghĩa nếu không có từ khóa khớp
    if 'process_category' not in filters:
        # Kiểm tra xem truy vấn có liên quan đến quy trình bằng vector search
        results = vector_store.similarity_search_with_score(
            query=processed_query,
            k=10,
            filter={'type': 'process_info'}
        )
        for doc, score in results:
            if score < 0.3:  # Ngưỡng tương đồng (cosine distance)
                category = doc.metadata.get('category', '')
                if category in process_keywords:
                    filters['process_category'] = category
                    break
    
    # Lấy toàn bộ tài liệu từ ChromaDB
    collection = vector_store._collection.get(include=["documents", "metadatas"])
    docs = [
        {
            "document": type('Document', (), {
                'page_content': doc,
                'metadata': meta
            })(),
            "price": float(meta.get('price_per_month', float('inf'))) if meta.get('price_per_month') else float('inf'),
            "distance": float('inf')
        }
        for doc, meta in zip(collection['documents'], collection['metadatas'])
    ]
    
    
    filtered_docs = []
    reject_reasons = {}
    
    for i, doc_item in enumerate(docs):
        doc = doc_item["document"]
        valid = True
        doc_reasons = []
        
        
        # Xử lý truy vấn quy trình
        if 'process_category' in filters:
            if doc.metadata.get('type') != 'process_info' or doc.metadata.get('category') != filters['process_category']:
                reason = f"Không phải tài liệu '{filters['process_category']}' (type={doc.metadata.get('type')}, category={doc.metadata.get('category')})"
                doc_reasons.append(reason)
                valid = False
        # Xử lý truy vấn phòng
        else:
            # Kiểm tra khoảng cách
            if 'location' in filters and 'latitude' in doc.metadata and 'longitude' in doc.metadata:
                try:
                    user_lat = filters['location']['latitude']
                    user_lon = filters['location']['longitude']
                    radius = filters['location']['radius']
                    doc_lat = float(doc.metadata['latitude'])
                    doc_lon = float(doc.metadata['longitude'])
                    distance = haversine_distance(user_lat, user_lon, doc_lat, doc_lon)
                    doc_item['distance'] = distance
                    if distance > radius:
                        reason = f"Khoảng cách {distance:.2f}km > bán kính {radius}km"
                        doc_reasons.append(reason)
                        valid = False
                except (ValueError, TypeError) as e:
                    print(f"[DEBUG] Lỗi tính khoảng cách: {str(e)}")
    
            # Kiểm tra giá
            if 'price_min' in filters and 'price_per_month' in doc.metadata:
                try:
                    doc_price = float(doc.metadata['price_per_month'])
                    if doc_price < filters['price_min']:
                        reason = f"Giá {doc_price:,}đ < giá tối thiểu {filters['price_min']:,}đ"
                        doc_reasons.append(reason)
                        valid = False
                except (ValueError, TypeError):
                    pass

            if 'price_max' in filters and 'price_per_month' in doc.metadata:
                try:
                    doc_price = float(doc.metadata['price_per_month'])
                    if doc_price > filters['price_max']:
                        reason = f"Giá {doc_price:,}đ > giá tối đa {filters['price_max']:,}đ"
                        doc_reasons.append(reason)
                        valid = False
                except (ValueError, TypeError):
                    pass
            
            # Kiểm tra vị trí quận/thành phố
            district_norm = normalize_location(doc.metadata.get('district_norm', doc.metadata.get('district', '')))
            city_norm = normalize_location(doc.metadata.get('city_norm', doc.metadata.get('city', '')))
            if (not district_norm or not city_norm) and 'address' in doc.metadata:
                extracted = extract_location_from_address(doc.metadata['address'])
                if not district_norm and 'district' in extracted:
                    district_norm = normalize_location(extracted['district'])
                if not city_norm and 'city' in extracted:
                    city_norm = normalize_location(extracted['city'])
            if 'district' in filters:
                        filter_district_raw = filters['district']
                        filter_district_norm = normalize_location(filter_district_raw)
                        filter_quan_match = re.search(r"quan(\d+)", filter_district_norm)
                        filter_quan_number = filter_quan_match.group(1) if filter_quan_match else None

                        # Lấy district từ metadata
                        doc_district_raw = doc.metadata.get('district', '')
                        doc_district_norm = normalize_location(doc_district_raw)
                        # Lấy address và chunk_content
                        address = doc.metadata.get('address', '')
                        chunk_content = doc.metadata.get('chunk_content', '')

                        found = False
                        # 1. So sánh trường district
                        if filter_quan_number and doc_district_norm == f"quan{filter_quan_number}":
                            found = True
                        # 2. So sánh trong address
                        elif filter_quan_number and contains_district_number(address, filter_quan_number):
                            found = True
                        # 3. So sánh trong chunk_content
                        elif filter_quan_number and contains_district_number(chunk_content, filter_quan_number):
                            found = True
                        # 4. Nếu không phải quận số, so sánh như cũ
                        elif not filter_quan_number and (doc_district_norm == filter_district_norm):
                            found = True

                        if not found:
                            reason = f"Không tìm thấy quận {filter_quan_number or filter_district_norm} trong tài liệu"
                            doc_reasons.append(reason)
                            valid = False

            if 'city' in filters:
                filter_city_norm = normalize_location(filters['city'])
                if not city_norm or (filter_city_norm not in city_norm and city_norm not in filter_city_norm):
                    reason = f"Thành phố '{filters['city']}' không khớp với '{doc.metadata.get('city', 'N/A')}'"
                    doc_reasons.append(reason)
                    valid = False

            # Kiểm tra loại nhà
            if 'type' in filters and 'type' in doc.metadata:
                filter_type = filters['type'].lower()
                doc_type = doc.metadata['type'].lower()
                matched = False
                for key, synonyms_list in synonyms.items():
                    if filter_type in synonyms_list and doc_type in synonyms_list:
                        matched = True
                        break
                if not matched and filter_type != doc_type:
                    reason = f"Loại nhà '{filter_type}' không khớp với '{doc_type}'"
                    doc_reasons.append(reason)
                    valid = False

            # Kiểm tra tiện ích
            if 'amenities' in filters:
                for amenity in filters['amenities']:
                    matched = False
                    if amenity == "đầy đủ nội thất" and 'full_furnishing' in doc.metadata:
                        if any(syn in doc.metadata['full_furnishing'].lower() for syn in amenities_synonyms[amenity]):
                            matched = True
                    if 'extensions' in doc.metadata:
                        if any(syn in doc.metadata['extensions'].lower() for syn in amenities_synonyms[amenity]):
                            matched = True
                    if any(syn in doc.page_content.lower() for syn in amenities_synonyms[amenity]):
                        matched = True
                    if not matched:
                        reason = f"Không có tiện ích '{amenity}'"
                        doc_reasons.append(reason)
                        valid = False

        if valid:
            filtered_docs.append(doc_item)
        else:
            reject_reasons[i] = doc_reasons
    
    # In số lượng tài liệu hợp lệ thực tế
    
    # Sắp xếp theo khoảng cách (nếu có tọa độ) hoặc giá (cho phòng), không sắp xếp cho quy trình
    if 'process_category' not in filters and 'location' in filters:
        filtered_docs.sort(key=lambda x: x.get("distance", float('inf')))
        
        # In khoảng cách của các kết quả sau khi sắp xếp
        for i, doc_item in enumerate(filtered_docs[:min(top_k, len(filtered_docs))]):
            doc = doc_item["document"]
            distance = doc_item.get("distance", float('inf'))
            distance_str = f"{distance:.2f}km" if distance != float('inf') else "Không rõ"
    elif 'process_category' not in filters:
        filtered_docs.sort(key=lambda x: x["price"])
    
    # Lấy top_k tài liệu
    filtered_docs = filtered_docs[:min(top_k, len(filtered_docs))]
    
    print("\n=== BÁO CÁO TÌM KIẾM ===")
    print(f"- Tổng số tài liệu tìm thấy: {len(docs)}")
    print(f"- Số tài liệu hợp lệ trả về: {len(filtered_docs)}")
    print(f"- Số tài liệu bị loại: {len(reject_reasons)}")
    
    if reject_reasons:
        for idx, reasons in list(reject_reasons.items())[:10]:
            doc_name = docs[idx]["document"].metadata.get('room_name', docs[idx]["document"].metadata.get('category', f'Tài liệu {idx+1}'))
    
    if filtered_docs:
        print("\n=== CHI TIẾT TÀI LIỆU HỢP LỆ ===")
        for i, doc_item in enumerate(filtered_docs):
            doc = doc_item["document"]
            if doc.metadata.get('type') == 'process_info':
                print(f"  category: {doc.metadata.get('category', 'N/A')}")
                print(f"  content: {doc.page_content}")
            else:
                print(f"  room_name: {doc.metadata.get('room_name', 'N/A')}")
                print(f"  price_per_month: {doc.metadata.get('price_per_month', 'N/A')}")
                print(f"  address: {doc.metadata.get('address', 'N/A')}")
                
                # Hiển thị khoảng cách nếu có
                if 'location' in filters and 'distance' in doc_item and doc_item['distance'] != float('inf'):
                    # Thêm thông tin khoảng cách vào metadata để hiển thị trong kết quả
                    doc.metadata['distance_km'] = f"{doc_item['distance']:.2f}"

    
    if not filtered_docs:
        print("Không tìm thấy kết quả phù hợp với điều kiện lọc.")
        return []
    
    return [{"document": doc_item["document"], "similarity": 0.0, "distance": doc_item.get("distance", None)} 
            for doc_item in filtered_docs]

if __name__ == "__main__":
    queries = [
        "Tìm nhà trọ ở quận 1 hồ chí minh giá dưới 7 triệu",
    ]
    for query in queries:
        try:
            print(f"\n=== Truy vấn: '{query}' ===\n")
            results = retrieve_documents(query, top_k=3)
            print(f"\nKết quả: {len(results)} tài liệu\n")
        except Exception as e:
            print(f"Lỗi khi xử lý truy vấn: {str(e)}")
            import traceback
            traceback.print_exc()
