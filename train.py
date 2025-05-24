import os
import json
import time
import argparse
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from retrieval import retrieve_documents, analyze_query



# Load biến môi trường từ file .env
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

class LLMTrainer:
    def __init__(self, output_dir="training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.models = {
            "gemini": {
                "provider": "openrouter",
                "model": "google/gemini-2.0-flash-exp:free"
            },
        }

        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(self.output_dir, f"llm_responses_{self.timestamp}.csv")
            
    def format_context_for_prompt(self, documents):
        context = []
        for i, doc in enumerate(documents):
            if "document" in doc:
                doc = doc["document"]

            metadata = doc.metadata
            content = doc.page_content

            if metadata.get('type') != 'process_info':
                context.append(f"--- Thông tin phòng {i+1} ---")
                context.append(f"Tên: {metadata.get('room_name', 'Không có tên')}")
                context.append(f"Địa chỉ: {metadata.get('address', 'Không có địa chỉ')}")
                context.append(f"Giá: {int(metadata.get('price_per_month', 0)):,}đ/tháng")
                if 'extensions' in metadata:
                    context.append(f"Tiện ích: {metadata.get('extensions', '')}")
                context.append(f"Link: {metadata.get('url', 'Không có link')}")
            else:
                context.append(f"--- Thông tin quy trình {metadata.get('category', 'chung')} ---")
                context.append(content)

            context.append("\n")
        return "\n".join(context)

    def create_prompt(self, query, documents):
        context = self.format_context_for_prompt(documents)
        analysis = analyze_query(query)

        if 'process_category' in analysis['filters']:
            # system_prompt = """
            #     Bạn là trợ lý ảo cho một website đăng tin cho thuê phòng trọ và căn hộ.
            #     Nhiệm vụ của bạn là giúp người dùng hiểu rõ các quy trình, hướng dẫn hoặc quy định liên quan (như thuê phòng, đăng tin, thanh toán, khiếu nại...).
            #     Hãy trả lời dựa trên thông tin ngữ cảnh cung cấp. Nếu thông tin không đầy đủ, hãy nêu những gì bạn biết và khuyên người dùng liên hệ website để được hỗ trợ thêm.

            #     Luôn trả lời bằng tiếng Việt. Trả lời rõ ràng, dễ hiểu và hữu ích, thân thiện.
            #     """
            system_prompt = """Bạn là trợ lý ảo cho một website đăng tin cho thuê phòng trọ và căn hộ.
                    Hãy trả lời ngắn gọn
                    Tạo phản hồi có cấu trúc rõ ràng, tập trung vào các thông tin quan trọng nhất.    
                    Nếu không có thông tin thì nêu khuyên người dùng liên hệ web để đựoc hỗ trợ thêm
                    Trả lời rõ ràng, dễ hiểu và hữu ích, thân thiện, đáng yêu."""
                
            user_prompt = f"""Ngữ cảnh:
{context}

Câu hỏi: {query}

Hãy giải thích quy trình một cách rõ ràng và dễ hiểu."""
        else:
            system_prompt = """Bạn là trợ lý ảo cho một website đăng tin cho thuê phòng trọ và căn hộ.
Nhiệm vụ của bạn là hỗ trợ người dùng tìm được phòng phù hợp nhất dựa trên thông tin ngữ cảnh (danh sách các phòng được đề xuất).
Nếu có nhiều kết quả, hãy tóm tắt các lựa chọn chính, so sánh ưu nhược điểm và đưa ra lời khuyên ngắn gọn chọn phòng. 
Nếu không có phòng phù hợp, hãy thông báo không có kết quả và đề xuất người dùng thay đổi tiêu chí tìm kiếm.Chèn link url phòng vào.
Luôn trả lời bằng tiếng Việt. Trả lời dễ hiểu ,ngắn gọn, đáng yêu (có các icon đáng yêu)."""
            user_prompt = f"""Ngữ cảnh:
{context}

Yêu cầu tìm phòng: {query}

Hãy tóm tắt các lựa chọn và đưa ra lời khuyên."""
        return system_prompt, user_prompt

 
    def get_openrouter_response(self, model_name, system_prompt, user_prompt, max_retries=3, backoff_time=10):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://homenest.com",
            "X-Title": "Rental Assistant"
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    print(f"Lỗi 429: Quá nhiều yêu cầu, đợi {backoff_time} giây rồi thử lại (lần {attempt+1}/{max_retries})")
                    time.sleep(backoff_time)
                else:
                    print(f"Lỗi API: {e}")
                    return f"Error: {str(e)}"
            except Exception as e:
                print(f"Lỗi không xác định: {e}")
                return f"Error: {str(e)}"
        return "Error: Đã thử tối đa số lần retry nhưng vẫn bị lỗi 429"

    def process_query(self, query, top_k=3):
        print(f"Xử lý truy vấn: '{query}'")
        documents = retrieve_documents(query, top_k=top_k)
        if not documents:
            print("Không tìm thấy tài liệu phù hợp")
            return

        system_prompt, user_prompt = self.create_prompt(query, documents)
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "responses": {}
        }

        for model_id, config in self.models.items():
            print(f"\n--- {model_id} ---")
            start_llm = time.time()
            response = self.get_openrouter_response(config["model"], system_prompt, user_prompt)
            results["responses"][model_id] = response
            time.sleep(1)

        self.results.append(results)
        self.save_results()
        return results

    def process_queries_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]

        for query in tqdm(queries, desc="Xử lý truy vấn"):
            self.process_query(query)

    def save_results(self):
        json_file = os.path.join(self.output_dir, f"llm_responses_{self.timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        rows = []
        for item in self.results:
            query = item["query"]
            timestamp = item["timestamp"]
            for model_id, response in item["responses"].items():
                rows.append({
                    "query": query,
                    "timestamp": timestamp,
                    "model": model_id,
                    "response": response
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_file, index=False, encoding='utf-8')
        print(f"\nĐã lưu kết quả vào {self.output_file}")

def main():
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--file", type=str)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--output", type=str, default="training_data")  # Nếu bạn cần thư mục output

    args = parser.parse_args()

    trainer = LLMTrainer(output_dir=args.output)

    if args.query:
        trainer.process_query(args.query, top_k=args.topk)
    elif args.file:
        trainer.process_queries_from_file(args.file)
    else:
        print("Vui lòng cung cấp --query hoặc --file")

if __name__ == "__main__":
    main() 
