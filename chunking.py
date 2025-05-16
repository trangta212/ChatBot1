import json
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("keepitreal/vietnamese-sbert")

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=256))

def split_content_semantic(text, max_tokens=256):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_tokens = count_tokens(para)
        current_tokens = count_tokens(current_chunk)
        
        if current_tokens + para_tokens <= max_tokens:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Nếu đoạn này quá dài, tiếp tục cắt nhỏ hơn ở mức câu
            if para_tokens > max_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                for sentence in sentences:
                    sent_tokens = count_tokens(sentence)
                    if count_tokens(temp_chunk) + sent_tokens <= max_tokens:
                        temp_chunk += sentence + " "
                    else:
                        if temp_chunk.strip():
                            chunks.append(temp_chunk.strip())
                        # Nếu câu vẫn quá dài, cắt ở mức từ
                        if sent_tokens > max_tokens:
                            words = sentence.split()
                            for i in range(0, len(words), 50):  # chunk 50 từ 1 lần
                                sub_chunk = " ".join(words[i:i+50])
                                chunks.append(sub_chunk.strip())
                            temp_chunk = ""
                        else:
                            temp_chunk = sentence + " "
                if temp_chunk.strip():
                    chunks.append(temp_chunk.strip())
                current_chunk = ""
            else:
                current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def clean_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) 
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    return text

def process_json_custom(input_path, output_path, max_tokens=256):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = []
    for item in data:
        room_name = item.get("room_name", "")
        address = item.get("address", "")
        price_per_month = item.get("price_per_month", None)
        area = item.get("area", None)
        extensions = ", ".join(item.get("extensions", [])) if isinstance(item.get("extensions", []), list) else item.get("extensions", "")
        full_furnishing = item.get("full_furnishing", "")
        start_date = item.get("start_date", "")
        expire = item.get("expire", "")
        latitude = item.get("latitude", None)
        longitude = item.get("longitude", None)
        type_ = item.get("type", "")
        combined_text = item.get("combined_text", "")
        url = item.get("url", "")
        
        if not combined_text.strip():
            continue
        
        split_parts = split_content_semantic(combined_text, max_tokens=max_tokens)
        for i, part in enumerate(split_parts):
            clean_part = clean_markdown(part).strip()
            if clean_part:
                chunks.append({
                    "room_name": room_name,
                    "address": address,
                    "price_per_month": price_per_month,
                    "area": area,
                    "extensions": extensions,
                    "full_furnishing": full_furnishing,
                    "start_date": start_date,
                    "expire": expire,
                    "latitude": latitude,
                    "longitude": longitude,
                    "type": type_,
                    "chunk_content": clean_part,
                    "chunk_id": i,
                    "url":url
                })
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(chunks, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_json_custom("preprocessed_roomInformation.json", "output_chunks.json", max_tokens=256)
