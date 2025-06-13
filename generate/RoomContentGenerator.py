import os
import json
import time
import argparse
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()
OPENROUTER_KEY = os.getenv("OPEN_GENERATE")

class RoomContentGenerator:
    """Tạo tiêu đề và mô tả hấp dẫn cho phòng trọ từ thông tin đầu vào"""
    
    def __init__(self, output_dir="generated_content"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.models = {
            "gemini": {
                "provider": "openrouter",
                "model": "google/gemini-2.0-flash-exp:free"
            },
            # Có thể thêm các model khác
            # "claude": {
            #     "provider": "openrouter", 
            #     "model": "anthropic/claude-3-haiku"
            # },
            # "gpt4": {
            #     "provider": "openrouter",
            #     "model": "openai/gpt-4o-mini"
            # }
        }
        
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_dir, f"room_content_{self.timestamp}.csv")
        
    def create_content_generation_prompt(self, room_info):
        """Tạo prompt để sinh tiêu đề và mô tả hấp dẫn"""
        
        system_prompt = """Bạn là một chuyên gia marketing bất động sản chuyên viết nội dung quảng cáo phòng trọ và căn hộ cho thuê.
Nhiệm vụ của bạn là tạo ra những tiêu đề và mô tả CỰC KỲ HẤP DẪN, lôi cuốn khách hàng thuê phòng.

Yêu cầu:
- Tiêu đề: Ngắn gọn (tối đa 80 ký tự), nổi bật, tạo cảm giác khẩn cấp hoặc độc đáo
- Mô tả: Chi tiết (150-300 từ), sinh động, tạo cảm xúc tích cực, nhấn mạnh lợi ích
- Sử dụng từ ngữ tích cực, emoji phù hợp
- Tạo cảm giác "không thể bỏ lỡ"
- Nhấn mạnh điểm mạnh, che đậy điểm yếu một cách khéo léo

Trả lời theo định dạng JSON:
{
    "title": "tiêu đề hấp dẫn",
    "description": "mô tả chi tiết và lôi cuốn"
}"""

        # Xử lý thông tin phòng
        room_details = []
        
        if room_info.get('room_name'):
            room_details.append(f"Tên phòng: {room_info['room_name']}")
        
        if room_info.get('room_type'):
            room_details.append(f"Loại phòng: {room_info['room_type']}")
            
        if room_info.get('area'):
            room_details.append(f"Diện tích: {room_info['area']}m²")
            
        if room_info.get('price'):
            try:
                price = float(room_info['price'])
                room_details.append(f"Giá thuê: {price:,.0f}đ/tháng")
            except (ValueError, TypeError):
                room_details.append(f"Giá thuê: {room_info['price']}")
            
        if room_info.get('address'):
            room_details.append(f"Địa chỉ: {room_info['address']}")
            
        if room_info.get('ward'):
            room_details.append(f"Phường/Xã: {room_info['ward']}")
            
        if room_info.get('district'):
            room_details.append(f"Quận/Huyện: {room_info['district']}")
            
        if room_info.get('city'):
            room_details.append(f"Thành phố: {room_info['city']}")
            
        if room_info.get('amenities'):
            amenities = room_info['amenities']
            if isinstance(amenities, list):
                amenities = ", ".join(amenities)
            room_details.append(f"Tiện nghi: {amenities}")
            
        if room_info.get('nearby_facilities'):
            facilities = room_info['nearby_facilities']
            if isinstance(facilities, list):
                facilities = ", ".join(facilities)
            room_details.append(f"Tiện ích xung quanh: {facilities}")
            
        if room_info.get('furniture'):
            furniture = room_info['furniture']
            if isinstance(furniture, list):
                furniture = ", ".join(furniture)
            room_details.append(f"Nội thất: {furniture}")
            
        if room_info.get('utilities'):
            utilities = room_info['utilities']
            if isinstance(utilities, list):
                utilities = ", ".join(utilities)
            room_details.append(f"Dịch vụ: {utilities}")
            
        if room_info.get('special_features'):
            features = room_info['special_features']
            if isinstance(features, list):
                features = ", ".join(features)
            room_details.append(f"Đặc điểm nổi bật: {features}")
            
        if room_info.get('target_tenant'):
            room_details.append(f"Phù hợp cho: {room_info['target_tenant']}")
            
        if room_info.get('deposit'):
            try:
                deposit = float(room_info['deposit'])
                room_details.append(f"Tiền cọc: {deposit:,.0f}đ")
            except (ValueError, TypeError):
                room_details.append(f"Tiền cọc: {room_info['deposit']}")
            
        if room_info.get('contract_duration'):
            room_details.append(f"Thời hạn thuê: {room_info['contract_duration']}")
            
        if room_info.get('move_in_date'):
            room_details.append(f"Ngày có thể vào ở: {room_info['move_in_date']}")
            
        # Thêm các trường khác nếu có
        for key, value in room_info.items():
            if key not in ['room_name', 'room_type', 'area', 'price', 'address', 
                          'ward', 'district', 'city', 'amenities', 'nearby_facilities',
                          'furniture', 'utilities', 'special_features', 'target_tenant',
                          'deposit', 'contract_duration', 'move_in_date'] and value:
                room_details.append(f"{key}: {value}")
            
        room_info_text = "\n".join(room_details)
        
        user_prompt = f"""Thông tin phòng trọ:
{room_info_text}

Hãy tạo tiêu đề và mô tả CỰC KỲ HẤP DẪN cho phòng này. Phải làm cho người đọc cảm thấy "WOW" và muốn thuê ngay lập tức!

Lưu ý khi viết:
- Nếu giá tốt (dưới 3 triệu) thì nhấn mạnh "GIÁ CỰC ƯU ĐÃI" 💰
- Nếu vị trí đẹp (gần trung tâm, trường học, bệnh viện) thì nhấn mạnh "VỊ TRÍ VÀNG" 📍
- Nếu diện tích lớn (trên 20m²) thì nhấn mạnh "RỘNG RÃII THOÁNG MÁT" 🏠
- Nếu tiện nghi đầy đủ thì nhấn mạnh "FULL NỘI THẤT, VÀO Ở NGAY" ✨
- Nếu an ninh tốt thì nhấn mạnh "AN TOÀN TUYỆT ĐỐI" 🔒
- Tạo cảm giác khan hiếm "SẮP HẾT PHÒNG", "CƠ HỘI CUỐI CÙNG" ⏰
- Sử dụng các từ khóa tích cực: sang trọng, hiện đại, tiện lợi, yên tĩnh

Trả lời bằng JSON như yêu cầu."""

        return system_prompt, user_prompt
    
    def get_openrouter_response(self, model_name, system_prompt, user_prompt, max_retries=3, backoff_time=10):
        """Gửi request tới OpenRouter API"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://homenest.com",
            "X-Title": "Room Content Generator"
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.8,  # Tăng creativity
            "max_tokens": 1000
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                
                # Thử parse JSON response
                try:
                    # Tìm JSON trong response (có thể có text khác xung quanh)
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = content[start_idx:end_idx]
                        parsed_json = json.loads(json_str)
                        
                        # Validate response
                        if "title" in parsed_json and "description" in parsed_json:
                            return parsed_json
                        else:
                            raise json.JSONDecodeError("Missing required fields", "", 0)
                    else:
                        # Nếu không tìm thấy JSON, trả về format mặc định
                        return {
                            "title": "🏠 Phòng đẹp giá tốt - Thuê ngay kẻo lỡ! ⏰",
                            "description": content[:500] + "..." if len(content) > 500 else content
                        }
                except json.JSONDecodeError:
                    return {
                        "title": "✨ Phòng trọ chất lượng - Giá cực hợp lý! 💰",
                        "description": content[:500] + "..." if len(content) > 500 else content
                    }
                    
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    print(f"⏰ Lỗi 429: Quá nhiều yêu cầu, đợi {backoff_time}s... (lần {attempt+1}/{max_retries})")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    print(f"❌ Lỗi API: {e}")
                    return {"title": "Error", "description": f"API Error: {str(e)}"}
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout, thử lại lần {attempt+1}/{max_retries}")
                time.sleep(2)
            except Exception as e:
                print(f"❌ Lỗi không xác định: {e}")
                return {"title": "Error", "description": f"Unknown Error: {str(e)}"}
                
        return {"title": "Error", "description": "Đã thử tối đa số lần retry nhưng vẫn thất bại"}

    def generate_content_for_room(self, room_info):
        """Tạo nội dung cho một phòng"""
        room_name = room_info.get('room_name', room_info.get('title', 'Không tên'))
        print(f"🏠 Tạo nội dung cho phòng: {room_name}")
        
        system_prompt, user_prompt = self.create_content_generation_prompt(room_info)
        
        result = {
            "room_info": room_info,
            "timestamp": datetime.now().isoformat(),
            "generated_content": {}
        }
        
        for model_id, config in self.models.items():
            print(f"🤖 Sử dụng model: {model_id}")
            start_time = time.time()
            content = self.get_openrouter_response(config["model"], system_prompt, user_prompt)
            end_time = time.time()
            
            result["generated_content"][model_id] = {
                "title": content.get("title", ""),
                "description": content.get("description", ""),
                "generation_time": end_time - start_time
            }
            
            print(f"✅ Hoàn thành trong {end_time - start_time:.2f}s")
            time.sleep(1)  # Tránh spam API
            
        self.results.append(result)
        self.save_results()
        return result
    
    def generate_content_from_csv(self, csv_file_path):
        """Tạo nội dung từ file CSV chứa thông tin phòng"""
        try:
            print(f"📁 Đọc file CSV: {csv_file_path}")
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            print(f"📊 Tìm thấy {len(df)} phòng trong file")
            
            for index, row in tqdm(df.iterrows(), total=len(df), desc="🏠 Tạo nội dung"):
                room_info = row.to_dict()
                # Xóa các giá trị NaN và None
                room_info = {k: v for k, v in room_info.items() 
                           if pd.notna(v) and v is not None and str(v).strip() != ''}
                self.generate_content_for_room(room_info)
                
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {csv_file_path}")
        except Exception as e:
            print(f"❌ Lỗi đọc file CSV: {e}")
    
    def generate_content_from_json(self, json_file_path):
        """Tạo nội dung từ file JSON chứa danh sách phòng"""
        try:
            print(f"📁 Đọc file JSON: {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                rooms_data = json.load(f)
                
            if isinstance(rooms_data, list):
                print(f"📊 Tìm thấy {len(rooms_data)} phòng trong file")
                for room_info in tqdm(rooms_data, desc="🏠 Tạo nội dung"):
                    self.generate_content_for_room(room_info)
            else:
                print("📊 Tạo nội dung cho 1 phòng")
                self.generate_content_for_room(rooms_data)
                
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {json_file_path}")
        except json.JSONDecodeError:
            print(f"❌ File JSON không hợp lệ: {json_file_path}")
        except Exception as e:
            print(f"❌ Lỗi đọc file JSON: {e}")
    
    def save_results(self):
        """Lưu kết quả vào file"""
        # Lưu JSON
        json_file = os.path.join(self.output_dir, f"generated_content_{self.timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Lưu CSV
        rows = []
        for item in self.results:
            room_info = item["room_info"]
            timestamp = item["timestamp"]
            
            for model_id, content in item["generated_content"].items():
                row = {
                    "timestamp": timestamp,
                    "model": model_id,
                    "room_name": room_info.get("room_name", ""),
                    "room_type": room_info.get("room_type", ""),
                    "area": room_info.get("area", ""),
                    "price": room_info.get("price", ""),
                    "address": room_info.get("address", ""), 
                    "district": room_info.get("district", ""),
                    "city": room_info.get("city", ""),
                    "generated_title": content.get("title", ""),
                    "generated_description": content.get("description", ""),
                    "generation_time": content.get("generation_time", 0)
                }
                
                # Thêm các trường khác từ room_info với prefix "original_"
                for key, value in room_info.items():
                    if key not in row and value:
                        row[f"original_{key}"] = value
                        
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.output_file, index=False, encoding='utf-8')
            print(f"💾 Đã lưu kết quả vào: {self.output_file}")
            print(f"📁 File JSON: {json_file}")

    def preview_content(self, room_info, model="gemini"):
        """Xem trước nội dung được tạo cho một phòng (không lưu file)"""
        print(f"👀 Xem trước nội dung cho phòng: {room_info.get('room_name', 'Không tên')}")
        
        system_prompt, user_prompt = self.create_content_generation_prompt(room_info)
        
        if model in self.models:
            content = self.get_openrouter_response(self.models[model]["model"], system_prompt, user_prompt)
            
            print("\n" + "="*60)
            print(f"📝 TIÊU ĐỀ: {content.get('title', '')}")
            print("="*60)
            print(f"📖 MÔ TẢ:\n{content.get('description', '')}")
            print("="*60)
            
            return content
        else:
            print(f"❌ Model '{model}' không được hỗ trợ. Các model có sẵn: {list(self.models.keys())}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Room Content Generator - Tạo tiêu đề và mô tả hấp dẫn cho phòng trọ")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=str, help="File CSV chứa thông tin phòng")
    group.add_argument("--json", type=str, help="File JSON chứa thông tin phòng")
    group.add_argument("--room-info", type=str, help="JSON string chứa thông tin một phòng")
    group.add_argument("--preview", type=str, help="Xem trước nội dung cho một phòng (JSON string)")
    
    # Options
    parser.add_argument("--output", type=str, default="generated_content", 
                       help="Thư mục lưu kết quả (mặc định: generated_content)")
    parser.add_argument("--model", type=str, default="gemini", 
                       help="Model sử dụng cho preview (mặc định: gemini)")

    args = parser.parse_args()

    # Tạo generator
    generator = RoomContentGenerator(output_dir=args.output)
    
    try:
        if args.csv:
            generator.generate_content_from_csv(args.csv)
        elif args.json:
            generator.generate_content_from_json(args.json)
        elif args.room_info:
            room_info = json.loads(args.room_info)
            generator.generate_content_for_room(room_info)
        elif args.preview:
            room_info = json.loads(args.preview)
            generator.preview_content(room_info, args.model)
            
    except json.JSONDecodeError:
        print("❌ Lỗi: JSON không hợp lệ")
    except KeyboardInterrupt:
        print("\n⏹️  Đã dừng bởi người dùng")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()