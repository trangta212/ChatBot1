# import json
# from datetime import datetime
# import re
# from collections import Counter

# ESSENTIAL_FIELDS = [
#     "room_name",
#     "address",
#     "price_per_month",
#     "area",
#     "description",
#     "extensions",
#     "full_furnishing",
#     "start_date",
#     "expire",
#     "latitude",
#     "longitude",
#     "type",
#     "url",
#     "lastName",
#     "phone_number",
#     "profile_picture",
#     "water_bill",
    
# ]

# # Danh sách các cụm từ mang tính PR cần loại bỏ
# PR_KEYWORDS = [
#     "siêu ưu đãi", "giá chỉ", "chỉ còn", "bao tất cả", "bao full", "đảm bảo", "miễn phí", "ưu đãi", 
#     "cực kỳ", "vô cùng", "trung tâm", "siêu rẻ", "siêu đẹp", "thoải mái", "tự do", "tiện nghi", 
#     "sang trọng", "cao cấp", "full nội thất", "rộng rãi", "an ninh", "sạch sẽ", "giờ giấc tự do", 
#     "vị trí đắc địa", "dịch vụ tốt", "nhiệt tình", "hỗ trợ", "quản lý 24/7"
# ]
# def preprocess_date(date_str):
#     """
#     Chuyển chuỗi date_str dạng "HH:MM dd/mm/yyyy" thành "yyyy-mm-dd" (chỉ lấy ngày).
#     Nếu không parse được trả về None.
#     """
#     try:
#         parts = date_str.split(",")[-1].strip()
#         dt = datetime.strptime(parts, "%H:%M %d/%m/%Y")
#         return dt.date().isoformat()  # chỉ lấy phần ngày
#     except:
#         return None


# def clean_text(text):
#     """
#     Làm sạch mô tả:
#     - Xoá ký hiệu, emoji, số điện thoại, email
#     - Bỏ các cụm từ mang tính PR
#     """
#     text = re.sub(r"[^\w\s,.–-]", "", text)
#     text = re.sub(r"[-=]{2,}", " ", text)
#     text = re.sub(r"[*]{2,}", " ", text)
#     text = re.sub(r"\b\d{8,11}\b", "", text)  # SĐT
#     text = re.sub(r"\S+@\S+", "", text)       # Email
#     text = re.sub(r"Liên hệ.*", "", text, flags=re.IGNORECASE)
#     text = re.sub(r"(ĐỊA CHỈ|HỆ THỐNG|BE HOME).*", "", text, flags=re.IGNORECASE)

#     # Bỏ cụm PR
#     for phrase in PR_KEYWORDS:
#         pattern = re.compile(re.escape(phrase), re.IGNORECASE)
#         text = pattern.sub("", text)

#     text = re.sub(r"\s{2,}", " ", text).strip()
#     return text

# def preprocess_room(room):
#     new_room = {}

#     # Xử lý start_date trước
#     start_date_str = room.get("start_date", None)
#     start_date = preprocess_date(start_date_str) if start_date_str else None

#     # Xử lý expire trước
#     expire_str = room.get("expire", None)
#     expire = preprocess_date(expire_str) if expire_str else None

#     for key in ESSENTIAL_FIELDS:
#         if key in room:
#             if key == "start_date":
#                 new_room[key] = start_date
#             elif key == "expire":
#                 # Nếu expire nhỏ hơn start_date hoặc None thì bỏ expire
#                 if expire and start_date and expire >= start_date:
#                     new_room[key] = expire
#                 else:
#                     # Bỏ trường expire nếu expire < start_date hoặc expire None
#                     pass
#             elif key == "area":
#                 area_str = str(room[key])
#                 match = re.search(r"(\d+)", area_str)
#                 new_room[key] = int(match.group(1)) if match else 0
#             else:
#                 new_room[key] = room[key]

#     # combined_text chỉ lấy từ description
#     # if "description" in new_room:
#     #     raw_text = "\n".join(new_room["description"])
#     #     combined_text = clean_text(raw_text)
#     # else:
#     #     combined_text = ""

#     # new_room["combined_text"] = combined_text
#     # new_room.pop("description", None)

#     return new_room

# # def get_top_keywords(rooms, top_n=20):
# #     all_words = []
# #     for room in rooms:
# #         text = room.get("combined_text", "")
# #         words = re.findall(r"\w+", text.lower())
# #         filtered = [w for w in words if len(w) > 2]
# #         all_words.extend(filtered)

# #     return Counter(all_words).most_common(top_n)

# # def get_top_keywords_by_type(rooms, top_n=10):
# #     type_word_map = {}

# #     for room in rooms:
# #         room_type = room.get("type", "unknown").lower()
# #         text = room.get("combined_text", "")
# #         words = re.findall(r"\w+", text.lower())
# #         filtered = [w for w in words if len(w) > 2]

# #         if room_type not in type_word_map:
# #             type_word_map[room_type] = []

# #         type_word_map[room_type].extend(filtered)

# #     result = {}
# #     for room_type, words in type_word_map.items():
# #         result[room_type] = Counter(words).most_common(top_n)

# #     return result

# def preprocess_all_rooms(input_file, output_file,
#                          top_keywords_file="top_keywords.json",
#                          type_keywords_file="top_keywords_by_type.json"):
#     with open(input_file, "r", encoding="utf-8") as f:
#         rooms = json.load(f)

#     processed = [preprocess_room(room) for room in rooms]

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(processed, f, ensure_ascii=False, indent=2)

#     print(f"✅ Tiền xử lý xong. {len(processed)} phòng được lưu vào {output_file}")

#     top_keywords = get_top_keywords(processed)
#     with open(top_keywords_file, "w", encoding="utf-8") as f:
#         json.dump(top_keywords, f, ensure_ascii=False, indent=2)

#     print(f"\n🔍 Top {len(top_keywords)} từ khóa phổ biến:")
#     for i, (word, freq) in enumerate(top_keywords, 1):
#         print(f"{i:2}. {word} ({freq})")

#     type_keywords = get_top_keywords_by_type(processed)
#     with open(type_keywords_file, "w", encoding="utf-8") as f:
#         json.dump(type_keywords, f, ensure_ascii=False, indent=2)

#     print("\n📊 Từ khóa phổ biến theo từng loại phòng:")
#     for room_type, keywords in type_keywords.items():
#         keyword_str = ", ".join([f"{word} ({count})" for word, count in keywords])
#         print(f"- {room_type}: {keyword_str}")

# if __name__ == "__main__":
#     preprocess_all_rooms(
#         input_file="cleaned_roomInformation.json",
#         output_file="preprocessed_roomInformation.json"
#     )
import json
from datetime import datetime, date
import re
from collections import Counter

ESSENTIAL_FIELDS = [
    "room_name",
    "address",
    "price_per_month",
    "area",
    "description",
    "extensions",
    "full_furnishing",
    "start_date",
    "expire",
    "latitude",
    "longitude",
    "type",
    "url",
    "lastName",
    "phone_number",
    "profile_picture",
    "water_bill",
    "electricity_bill",
    "room_images"
]

# Danh sách các cụm từ mang tính PR cần loại bỏ
PR_KEYWORDS = [
    "siêu ưu đãi", "giá chỉ", "chỉ còn", "bao tất cả", "bao full", "đảm bảo", "miễn phí", "ưu đãi", 
    "cực kỳ", "vô cùng", "trung tâm", "siêu rẻ", "siêu đẹp", "thoải mái", "tự do", "tiện nghi", 
    "sang trọng", "cao cấp", "full nội thất", "rộng rãi", "an ninh", "sạch sẽ", "giờ giấc tự do", 
    "vị trí đắc địa", "dịch vụ tốt", "nhiệt tình", "hỗ trợ", "quản lý 24/7"
]

def preprocess_date(date_str):
    """
    Chuyển chuỗi date_str dạng "HH:MM dd/mm/yyyy" thành đối tượng datetime.date.
    Nếu không parse được trả về None.
    """
    if not date_str:
        return None
    try:
        parts = date_str.split(",")[-1].strip()
        dt = datetime.strptime(parts, "%H:%M %d/%m/%Y")
        return dt.date()  # Trả về đối tượng date, chưa isoformat
    except Exception:
        return None

def replace_year_safe(date_obj, new_year):
    """
    Thay đổi năm của date_obj sang new_year.
    Nếu date_obj là ngày 29/2 và new_year không phải năm nhuận,
    sẽ chuyển ngày thành 28/2 để tránh lỗi.
    """
    if not isinstance(date_obj, (date, datetime)):
        return date_obj
    try:
        return date_obj.replace(year=new_year)
    except ValueError:
        # Xử lý trường hợp ngày 29/2 không tồn tại trong năm mới
        if date_obj.month == 2 and date_obj.day == 29:
            return date_obj.replace(year=new_year, day=28)
        else:
            raise

def clean_text(text):
    """
    Làm sạch mô tả:
    - Xoá ký hiệu, emoji, số điện thoại, email
    - Bỏ các cụm từ mang tính PR
    """
    text = re.sub(r"[^\w\s,.–-]", "", text)
    text = re.sub(r"[-=]{2,}", " ", text)
    text = re.sub(r"[*]{2,}", " ", text)
    text = re.sub(r"\b\d{8,11}\b", "", text)  # SĐT
    text = re.sub(r"\S+@\S+", "", text)       # Email
    text = re.sub(r"Liên hệ.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(ĐỊA CHỈ|HỆ THỐNG|BE HOME).*", "", text, flags=re.IGNORECASE)

    # Bỏ cụm PR
    for phrase in PR_KEYWORDS:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub("", text)

    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def preprocess_room(room):
    new_room = {}

    start_date_str = room.get("start_date", None)
    expire_str = room.get("expire", None)

    start_date = preprocess_date(start_date_str)
    expire = preprocess_date(expire_str)

    for key in ESSENTIAL_FIELDS:
        if key in room:
            if key == "start_date":
                if isinstance(start_date, (date, datetime)):
                    new_room[key] = start_date.isoformat()
                else:
                    new_room[key] = None

            elif key == "expire":
                if expire and start_date and expire >= start_date:
                    expire_2026 = replace_year_safe(expire, 2026)
                    new_room[key] = expire_2026.isoformat()
                else:
                    # Bỏ expire nếu không hợp lệ
                    pass

            elif key == "area":
                area_str = str(room[key])
                match = re.search(r"(\d+)", area_str)
                new_room[key] = int(match.group(1)) if match else 0

            else:
                new_room[key] = room[key]

    # Nếu muốn, có thể xử lý description để tạo combined_text ở đây
    # Ví dụ:
    # if "description" in new_room:
    #     combined_text = clean_text(new_room["description"])
    #     new_room["combined_text"] = combined_text

    return new_room

def preprocess_all_rooms(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        rooms = json.load(f)

    processed = [preprocess_room(room) for room in rooms]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"✅ Tiền xử lý xong. {len(processed)} phòng được lưu vào {output_file}")

if __name__ == "__main__":
    preprocess_all_rooms(
        input_file="cleaned_roomInformation.json",
        output_file="roomInfor.json"
    )
