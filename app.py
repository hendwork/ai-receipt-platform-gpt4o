# ==============================================================================
# 1. IMPORT LIBRARY
# ==============================================================================
import os, re, base64, json, uuid
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Import modul lokal, pastikan file vector_db.py ada
from vector_db import SimpleVectorDB

# ==============================================================================
# 2. INISIALISASI & KONFIGURASI
# ==============================================================================
load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Inisialisasi Model, DB, & Client API (Dilakukan Sekali) ---
print("Memuat model embedding & inisialisasi...")
# Baris ini akan mengunduh model saat pertama kali dijalankan. Harap bersabar.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
vector_db = SimpleVectorDB()
receipt_database = {}
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def image_to_base64(image_path):
    """Mengubah file gambar menjadi string base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==============================================================================
# 3. ROUTES / ENDPOINTS
# ==============================================================================
@app.route('/')
def index():
    """Menampilkan halaman utama UI."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_receipt():
    """Menerima, menganalisis gambar dengan GPT-4o, dan menyimpan hasilnya."""
    if 'receipt' not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400
    file = request.files['receipt']
    if not file or not file.filename:
        return jsonify({"error": "File tidak valid"}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        base64_image = image_to_base64(filepath)
        prompt_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analisis gambar struk ini. Ekstrak semua item yang dibeli dan harganya. Kembalikan data HANYA sebagai objek JSON yang valid, tanpa teks atau penjelasan lain. JSON harus memiliki satu kunci \"items\" yang merupakan array objek. Setiap objek harus memiliki dua kunci: \"name\" (string) dan \"price\" (integer, dalam unit mata uang terkecil, contoh: Rupiah). Contoh: {\"items\": [{\"name\": \"Burger Sapi\", \"price\": 55000}]}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=prompt_messages,
            max_tokens=1000
        )
        
        raw_response_str = response.choices[0].message.content
        
        # Parser JSON yang kuat untuk mengekstrak blok JSON
        json_match = re.search(r'\{.*\}', raw_response_str, re.DOTALL)
        if not json_match:
            print(f"Gagal menemukan JSON dalam respons AI: {raw_response_str}")
            return jsonify({"error": "AI tidak mengembalikan format data yang valid."}), 500
            
        json_str = json_match.group(0)
        extracted_data = json.loads(json_str)
        items = extracted_data.get("items", [])
        
        if not items:
            return jsonify({"error": "GPT-4o tidak dapat menemukan item pada struk."}), 500
            
        receipt_id = f"receipt_{len(receipt_database) + 1}"
        receipt_data = {
            "timestamp": datetime.now(),
            "items": [{"id": f"item_{uuid.uuid4()}", **item} for item in items],
            "total": sum(item['price'] for item in items)
        }
        receipt_database[receipt_id] = receipt_data
        
        for item in receipt_data["items"]:
            vector = embedding_model.encode(item['name'])
            metadata = {"receipt_id": receipt_id, "name": item['name'], "price": item['price'], "date": receipt_data['timestamp'].strftime("%Y-%m-%d")}
            vector_db.add_item(item['id'], vector, metadata)
            
        return jsonify({"message": "Struk diproses oleh GPT-4o!", "receipt_id": receipt_id, "data": receipt_data})

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}. Respons mentah dari AI: {raw_response_str}")
        return jsonify({"error": "Gagal mem-parsing respons dari AI."}), 500
    except Exception as e:
        print(f"OpenAI Vision Error: {e}")
        return jsonify({"error": f"Terjadi kesalahan pada model Vision AI: {e}"}), 500

@app.route('/ask', methods=['POST'])
def ask_ai():
    """Menjawab pertanyaan pengguna berdasarkan data yang sudah disimpan."""
    user_question = request.json.get('question', '').lower()
    if not user_question:
        return jsonify({"error": "Pertanyaan kosong"}), 400
        
    context = "Tidak ada informasi relevan yang ditemukan."
    
    if "total" in user_question or "expenses" in user_question:
        match = re.search(r'(\d{1,2} \w+)', user_question)
        if match:
            try:
                target_date = datetime.strptime(f"{match.group(1)} {datetime.now().year}", "%d %B").date()
                total = sum(r['total'] for r in receipt_database.values() if r['timestamp'].date() == target_date)
                context = f"Total pengeluaran pada tanggal {target_date.strftime('%d %B')} adalah Rp {total:,}."
            except ValueError:
                context = "Format tanggal tidak dikenali. Gunakan '20 June'."
        else:
            total = sum(r['total'] for r in receipt_database.values())
            context = f"Total pengeluaran dari semua struk adalah Rp {total:,}."
    elif "yesterday" in user_question:
        yesterday = (datetime.now() - timedelta(days=1)).date()
        items_bought = [item['name'] for r in receipt_database.values() if r['timestamp'].date() == yesterday for item in r['items']]
        context = f"Kemarin Anda membeli: {', '.join(items_bought)}." if items_bought else "Anda tidak membeli apa-apa kemarin."
    else:
        search_term = user_question.replace("where did i buy", "").replace("from last 7 day", "").strip()
        query_vector = embedding_model.encode(search_term)
        similar_items = vector_db.find_similar_items(query_vector, k=3)
        if similar_items:
            found_items_info = []
            for _, score, meta in similar_items:
                is_within_7_days = (datetime.now().date() - datetime.strptime(meta['date'], "%Y-%m-%d").date()).days <= 7
                if "last 7 day" in user_question and not is_within_7_days:
                    continue
                info = f"'{meta['name']}' (dibeli pada {meta['date']}, kemiripan: {score:.2f})"
                found_items_info.append(info)
            if found_items_info:
                context = f"Berikut adalah item yang mirip dengan '{search_term}' yang ditemukan: {'; '.join(found_items_info)}."
                
    try:
        prompt = f"Anda adalah asisten keuangan yang ramah. Jawab pertanyaan pengguna HANYA berdasarkan konteks yang diberikan. Buat jawaban singkat dalam Bahasa Indonesia.\nKonteks: {context}\nPertanyaan Pengguna: {user_question}\nJawaban:"
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return jsonify({"error": f"Error saat menghubungi OpenAI: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)