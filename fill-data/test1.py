import json

# Đọc dữ liệu JSON gốc
with open("grouped_users.json", "r", encoding="utf-8") as f:
    data = json.load(f)

simplified_rooms = []

for user in data["users"]:
    for room in user["rooms"]:
        simplified = {
            "room_name": room["room_name"],
            "address": room["address"],
            "price_per_month": room["price_per_month"],
            "area": room["area"],
            "extensions": room["extensions"],
            "full_furnishing": "Đầy đủ nội thất" if room["full_furnishing"] == 1 else "Không có nội thất",
            "start_date": room["start_date"],
            "expire": room["expire"],
            "latitude": room["latitude"],
            "longitude": room["longitude"],
            "type": room["type"],
            "combined_text": "\n".join(room["description"])
        }
        simplified_rooms.append(simplified)

# Ghi ra file JSON mới
with open("simplified_rooms.json", "w", encoding="utf-8") as f:
    json.dump(simplified_rooms, f, ensure_ascii=False, indent=2)
