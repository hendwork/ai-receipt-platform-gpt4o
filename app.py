import os
import re
import base64
import json
import uuid
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from vector_db import SimpleVectorDB

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("Mohon tunggu! Sedang memanaskan mesin AI...")

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_database = SimpleVectorDB()
receipts_storage = {}
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def convert_image_to_base64_string(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def process_receipt_upload():
    if 'receipt' not in request.files:
        return jsonify({"error": "Anda lupa melampirkan file struk."}), 400
    
    uploaded_file = request.files['receipt']
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "File yang diunggah sepertinya tidak valid."}), 400
        
    saved_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(saved_image_path)
    
    try:
        image_in_base64 = convert_image_to_base64_string(saved_image_path)
        
        gpt4o_vision_prompt = [{
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Tolong analisis gambar struk ini. Ekstrak setiap item yang dibeli beserta harganya. "
                        "Sangat penting: Kembalikan hasilnya HANYA sebagai objek JSON. Jangan tambahkan teks atau "
                        "penjelasan apa pun di luar JSON. JSON harus punya satu kunci utama 'items', "
                        "yang berisi array objek. Setiap objek item harus punya kunci 'name' (string) dan 'price' (integer, dalam Rupiah). "
                        "Contoh format yang benar: {\"items\": [{\"name\": \"Kopi Susu\", \"price\": 18000}]}"
                    )
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{image_in_base64}"}
                }
            ]
        }]
        
        api_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=gpt4o_vision_prompt,
            max_tokens=1000
        )
        
        raw_ai_response = api_response.choices[0].message.content
        
        json_block_match = re.search(r'\{.*\}', raw_ai_response, re.DOTALL)
        if not json_block_match:
            print(f"Gagal menemukan JSON di respons AI: {raw_ai_response}")
            return jsonify({"error": "Format respons dari AI tidak valid."}), 500
            
        json_data_string = json_block_match.group(0)
        extracted_data = json.loads(json_data_string)
        items = extracted_data.get("items", [])
        
        if not items:
            return jsonify({"error": "Maaf, AI tidak dapat menemukan item apa pun di struk ini."}), 500
            
        new_receipt_id = f"struk_{len(receipts_storage) + 1}"
        new_receipt_data = {
            "timestamp": datetime.now(),
            "items": [{"id": f"item_{uuid.uuid4()}", **item} for item in items],
            "total": sum(item['price'] for item in items)
        }
        receipts_storage[new_receipt_id] = new_receipt_data
        
        for item in new_receipt_data["items"]:
            item_name_vector = sentence_model.encode(item['name'])
            metadata = {
                "receipt_id": new_receipt_id, 
                "name": item['name'], 
                "price": item['price'], 
                "date": new_receipt_data['timestamp'].strftime("%Y-%m-%d")
            }
            vector_database.add_item(item['id'], item_name_vector, metadata)
            
        return jsonify({
            "message": "Struk berhasil dianalisis oleh GPT-4o!", 
            "receipt_id": new_receipt_id, 
            "data": new_receipt_data
        })

    except json.JSONDecodeError:
        print(f"Error saat mem-parsing JSON. Respons mentah dari AI: {raw_ai_response}")
        return jsonify({"error": "Gagal membaca format JSON dari respons AI."}), 500
    except Exception as e:
        print(f"Terjadi kesalahan tak terduga: {e}")
        return jsonify({"error": f"Terjadi kesalahan pada sistem AI: {e}"}), 500

@app.route('/ask', methods=['POST'])
def handle_user_question():
    question_from_user = request.json.get('question', '').lower()
    if not question_from_user:
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400
        
    information_context = "Maaf, saya tidak dapat menemukan informasi yang relevan untuk menjawab itu."
    
    if "total" in question_from_user or "pengeluaran" in question_from_user:
        date_match = re.search(r'(\d{1,2} \w+)', question_from_user)
        if date_match:
            try:
                target_date = datetime.strptime(f"{date_match.group(1)} {datetime.now().year}", "%d %B").date()
                total_on_date = sum(r['total'] for r in receipts_storage.values() if r['timestamp'].date() == target_date)
                information_context = f"Total pengeluaran pada tanggal {target_date.strftime('%d %B')} adalah Rp {total_on_date:,}."
            except ValueError:
                information_context = "Format tanggal tidak saya kenali. Coba gunakan format seperti '20 Juni'."
        else:
            grand_total = sum(r['total'] for r in receipts_storage.values())
            information_context = f"Total pengeluaran dari semua struk yang tersimpan adalah Rp {grand_total:,}."

    elif "kemarin" in question_from_user:
        yesterday_date = (datetime.now() - timedelta(days=1)).date()
        items_bought_yesterday = [
            item['name'] 
            for r in receipts_storage.values() 
            if r['timestamp'].date() == yesterday_date 
            for item in r['items']
        ]
        if items_bought_yesterday:
            information_context = f"Berdasarkan data, kemarin Anda membeli: {', '.join(items_bought_yesterday)}."
        else:
            information_context = "Tidak ada data pembelian untuk kemarin."
            
    else:
        search_term = question_from_user.replace("di mana saya beli", "").replace("dalam 7 hari terakhir", "").strip()
        query_vector = sentence_model.encode(search_term)
        
        similar_items = vector_database.find_similar_items(query_vector, k=3)
        
        if similar_items:
            found_items_info = []
            for _, score, meta in similar_items:
                is_within_7_days = (datetime.now().date() - datetime.strptime(meta['date'], "%Y-%m-%d").date()).days <= 7
                if "7 hari terakhir" in question_from_user and not is_within_7_days:
                    continue
                
                info = f"'{meta['name']}' (dibeli pada {meta['date']}, tingkat kemiripan: {score:.2f})"
                found_items_info.append(info)
                
            if found_items_info:
                information_context = f"Saya menemukan beberapa item yang mirip dengan '{search_term}': {'; '.join(found_items_info)}."
                
    try:
        final_prompt = (
            "Anda adalah asisten keuangan yang cerdas dan ramah. "
            "Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan informasi dalam 'Konteks' yang saya berikan. "
            "Jawablah dengan singkat, jelas, dan dalam Bahasa Indonesia.\n\n"
            f"Konteks: {information_context}\n"
            f"Pertanyaan Pengguna: {question_from_user}\n\n"
            "Jawaban Anda:"
        )
        
        response_from_ai = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": final_prompt}]
        )
        final_answer = response_from_ai.choices[0].message.content
        return jsonify({"answer": final_answer})
        
    except Exception as e:
        print(f"Error saat menghubungi OpenAI untuk jawaban akhir: {e}")
        return jsonify({"error": f"Terjadi kesalahan saat menghasilkan jawaban: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)