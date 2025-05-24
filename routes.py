from flask import Blueprint, request, jsonify
import subprocess
import os
import json
import time
from datetime import datetime
from generate.RoomContentGenerator import RoomContentGenerator

routes = Blueprint('routes', __name__)

def run_train_query(query, top_k=3):  # Mặc định top_k=3
    train_py_path = '/Users/trangta/Documents/ChatBot/train.py'
    output_dir = 'training_data'
    command = f"python {train_py_path} --query \"{query}\" --topk {top_k} --output {output_dir}"
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error running train.py: {process.stderr}")
        return None
    
    files = [f for f in os.listdir(output_dir) if f.startswith('llm_responses_') and f.endswith('.json')]
    if not files:
        print("No result files found")
        return None
    latest_file = sorted(files)[-1]
    file_path = os.path.join(output_dir, latest_file)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if results:
        latest_result = results[-1]
        response = latest_result["responses"]["gemini"]
        return {
            "query": latest_result["query"],
            "response": response,
            "response_time": time.time() - datetime.fromisoformat(latest_result["timestamp"].replace('Z', '+00:00')).timestamp()
        }
    return None

@routes.route('/llm', methods=['POST'])
def llm():
    data = request.json
    query = data.get('query')
    
    print(f"Query nhận được từ FE lúc {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}: '{query}'")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    result = run_train_query(query, top_k=3)  # Mặc định top_k=3
    
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "No response from train.py"}), 500

# Khởi tạo RoomContentGenerator
room_generator = RoomContentGenerator()

@routes.route('/api/generate-room-content', methods=['POST'])
def generate_room_content():
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Không có dữ liệu được gửi lên'
            }), 400
            
        # Validate dữ liệu bắt buộc
        required_fields = ['room_type', 'price', 'address']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Thiếu các trường bắt buộc: {", ".join(missing_fields)}'
            }), 400
            
        # Tạo nội dung cho phòng
        result = room_generator.generate_content_for_room(data)
        
        if not result or 'generated_content' not in result:
            return jsonify({
                'success': False,
                'error': 'Không thể tạo nội dung cho phòng'
            }), 500
            
        # Trả về kết quả
        return jsonify({
            'success': True,
            'data': {
                'title': result['generated_content'].get('gemini', {}).get('title', ''),
                'description': result['generated_content'].get('gemini', {}).get('description', ''),
                'timestamp': result.get('timestamp', '')
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi server: {str(e)}'
        }), 500

