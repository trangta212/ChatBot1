import json
import random

# Danh sách họ, tên đệm và tên phổ biến Việt Nam
ho_viet = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Phan", "Vũ", "Đặng", "Bùi", "Đỗ"]
ten_dem = ["Văn", "Thị", "Hữu", "Minh", "Ngọc", "Quang", "Thu", "Thanh", "Tuấn", "Xuân"]
ten = ["An", "Bình", "Cường", "Dũng", "Hạnh", "Hoa", "Hưng", "Linh", "Mai", "Nam", "Oanh", "Phúc", "Quỳnh", "Sơn", "Trang", "Vy"]

def gen_random_name():
    return f"{random.choice(ho_viet)} {random.choice(ten_dem)} {random.choice(ten)}"

def gen_random_phone():
    # Đầu số phổ biến 09x hoặc 03x
    dau_so = random.choice(["090", "091", "093", "097", "098", "032", "033", "034", "035", "036", "037", "038", "039"])
    # Thêm 7 số còn lại (0-9)
    so_con_lai = "".join(random.choices("0123456789", k=7))
    return dau_so + so_con_lai

def fill_missing_fields(room):
    if (room.get("lastName", "").strip() == "" and
        room.get("phone_number", "").strip() == "" and
        room.get("profile_picture", "").strip() == ""):
        
        room["lastName"] = gen_random_name()
        room["phone_number"] = gen_random_phone()
        room["profile_picture"] = "https://phongtro123.com/images/default-user.svg"
    return room

def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        data = [fill_missing_fields(room) for room in data]
    else:
        print("File JSON không phải là list, vui lòng kiểm tra định dạng.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã xử lý và lưu file mới vào {output_file}")

if __name__ == "__main__":
    process_file("roomInfor.json", "roomInfor_filled.json")
