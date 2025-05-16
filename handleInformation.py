import json
import re
import unicodedata

# ==========================
# Utility Functions
# ==========================

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text.strip().replace("\xa0", " "))


def remove_duplicates(lst):
    return list(dict.fromkeys(lst))


def parse_price(price_str):
    match = re.search(r"([\d.,]+)", price_str.replace(",", "."))
    if match:
        try:
            return int(float(match.group(1)) * 1_000_000)  # VND
        except:
            return 0
    return 0


def parse_area(area_str):
    match = re.search(r"(\d+)", area_str)
    return int(match.group(1)) if match else 0


# ==========================
# Cleaning Logic per Entry
# ==========================

def clean_json_entry(entry):
    cleaned = {}

    # Normalize simple fields
    fields_to_normalize = [
        "room_name", "address", "price_per_month", "area", "start_date",
        "expire", "full_furnishing", "type", "lastName", "phone_number"
    ]
    for field in fields_to_normalize:
     if field == "price_per_month":
        cleaned[field] = parse_price(entry.get(field, "0"))
     else:
        cleaned[field] = normalize_text(entry.get(field, ""))

     # Normalize and remove duplicates in description & extensions
    cleaned["description"] = remove_duplicates([
        normalize_text(item) for item in entry.get("description", []) if isinstance(item, str)
    ])

    extensions = remove_duplicates([
        normalize_text(ext) for ext in entry.get("extensions", []) if isinstance(ext, str)
    ])
    cleaned["extensions"] = extensions if extensions else ["Không có tiện ích mở rộng"]

    # Parse bills
    try:
        cleaned["electricity_bill"] = int(entry.get("electricity_bill", 0))
    except:
        cleaned["electricity_bill"] = 0

    try:
        cleaned["water_bill"] = int(entry.get("water_bill", 0))
    except:
        cleaned["water_bill"] = 0

    # Giữ lại room_images (để xử lý sau ở bước tiền xử lý)
    cleaned["room_images"] = remove_duplicates(entry.get("room_images", []))

    # Giữ lại profile_picture để người dùng xử lý sau nếu muốn
    cleaned["profile_picture"] = entry.get("profile_picture", "")

    # Giữ lại tọa độ như bạn yêu cầu
    try:
        cleaned["latitude"] = float(entry.get("latitude", 0))
        cleaned["longitude"] = float(entry.get("longitude", 0))
    except:
        cleaned["latitude"] = 0.0
        cleaned["longitude"] = 0.0

    return cleaned


# ==========================
# Filters
# ==========================

REQUIRED_FIELDS = ["room_name", "address", "price_per_month", "description","room_images", "start_date", "expire"]

def has_required_fields(room):
    for field in REQUIRED_FIELDS:
        if not room.get(field):
            return False
    return True


def is_duplicate_room(room1, room2):
    return (
        room1["address"].lower() == room2["address"].lower() and
        room1["room_name"].lower() == room2["room_name"].lower()
    )


def remove_duplicate_rooms(rooms):
    unique = []
    for room in rooms:
        if not any(is_duplicate_room(room, r) for r in unique):
            unique.append(room)
    return unique


# ==========================
# Main Pipeline
# ==========================

def clean_all_rooms(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned = []
    for raw_room in raw_data:
        room = clean_json_entry(raw_room)

        if not has_required_fields(room):
            continue

        cleaned.append(room)

    final_data = remove_duplicate_rooms(cleaned)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Cleaned {len(final_data)} valid rooms saved to {output_file}")


# ==========================
# Run Script
# ==========================

if __name__ == "__main__":
    clean_all_rooms("roomInformation.json", "cleaned_roomInformation.json")
