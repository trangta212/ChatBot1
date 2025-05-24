import json

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
    "lastName",
    "phone_number",
    "profile_picture",
    "water_bill",
    "electricity_bill",
    "room_images"
]

def check_empty_fields(input_file, report_file):
    with open(input_file, "r", encoding="utf-8") as f:
        rooms = json.load(f)

    empty_report = []
    for i, room in enumerate(rooms, start=1):
        empty_fields = []
        for field in ESSENTIAL_FIELDS:
            # Kiểm tra trường không tồn tại hoặc rỗng (chuỗi trống hoặc None)
            if field not in room or room[field] in [None, "", [], {}]:
                empty_fields.append(field)
        if empty_fields:
            empty_report.append((i, empty_fields))

    with open(report_file, "w", encoding="utf-8") as f:
        if not empty_report:
            f.write("✅ Không có trường trống nào trong tất cả các phòng.\n")
            print("✅ Không có trường trống nào trong tất cả các phòng.")
        else:
            for idx, fields in empty_report:
                line = f"Phòng số {idx} thiếu hoặc trống các trường: {', '.join(fields)}\n"
                f.write(line)
                print(line.strip())

if __name__ == "__main__":
    input_json = "roomInfor_filled.json"           # Thay bằng tên file JSON của bạn
    report_txt = "empty_fields_report.txt" # File kết quả báo cáo

    check_empty_fields(input_json, report_txt)
