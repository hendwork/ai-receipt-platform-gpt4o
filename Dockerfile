# Langkah 1: Mulai dari sistem operasi dasar yang sudah memiliki Python 3.9 yang ringan.
FROM python:3.9-slim

# Catatan: Kita tidak perlu menginstal Tesseract OCR karena GPT-4o menanganinya.

# Langkah 2: Buat dan tentukan direktori kerja di dalam kontainer.
WORKDIR /app

# Langkah 3: Salin file daftar kebutuhan (requirements.txt) terlebih dahulu.
# Ini adalah trik optimisasi: Docker hanya akan menginstal ulang library jika file ini berubah.
COPY requirements.txt .

# Langkah 4: Jalankan perintah pip untuk menginstal semua library Python dari daftar.
RUN pip install --no-cache-dir -r requirements.txt

# Langkah 5: Salin semua sisa file proyek (app.py, vector_db.py, folder templates, dll.) ke dalam direktori kerja.
COPY . .

# Langkah 6: Beri tahu Docker bahwa aplikasi di dalam kontainer ini akan berjalan di port 5000.
EXPOSE 5000

# Langkah 7: Perintah terakhir yang akan dijalankan saat kontainer dimulai.
CMD ["python", "app.py"]