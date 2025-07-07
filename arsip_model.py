import streamlit as st
import mysql.connector
import pandas as pd
import base64
import time
import uuid
from datetime import datetime

def connect_to_db():
    return mysql.connector.connect(
        host="sql12.freesqldatabase.com",
        user="sql12788718", 
        password="zh24wGnPJN", 
        database="sql12788718" 
    )

def generate_kode_model(prefix="ML"):
    unique_id = uuid.uuid4().hex[:6].upper()
    return f"{prefix}.{unique_id}"

def download_button(file_blob, filename):
    b64 = base64.b64encode(file_blob).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{filename}</a>'
    return href

def main(sort_option="Terbaru", search_query=""):
    db = connect_to_db()
    cursor = db.cursor(dictionary=True)
    current_user_id = st.session_state.user["id_pengguna"]

    cursor.execute("""
        SELECT * FROM arsip_models
        WHERE id_pengguna = %s
        ORDER BY tanggal_dibuat DESC
    """, (current_user_id,))
    data = cursor.fetchall()
    cursor.close()
    db.close()
    st.markdown("<div style='padding-top: 1.2rem;'>", unsafe_allow_html=True)
    st.subheader("Arsip Model Machine Learning Anda")

    if not data:
        st.info("📭 Tidak ada model dalam arsip.")
        return

    data_model = pd.DataFrame(data)
    data_model['tanggal_dibuat'] = pd.to_datetime(data_model['tanggal_dibuat'], errors='coerce')

    # Urutan
    if sort_option == "Kode":
        data_model = data_model.sort_values(by="kode_model", ascending=True)
    elif sort_option == "Algoritma":
        data_model = data_model.sort_values(by="nama_algoritma", ascending=True)
    elif sort_option == "Terbaru":
        data_model = data_model.sort_values(by="tanggal_dibuat", ascending=False)
    elif sort_option == "Terlama":
        data_model = data_model.sort_values(by="tanggal_dibuat", ascending=True)
    elif sort_option == "Pembuat":
        data_model = data_model.sort_values(by="nama_pengguna", ascending=True)

    data = data_model.to_dict(orient="records")

    search_query = st.text_input("🔎 Cari Model", placeholder="Masukkan nama algoritma atau kode model")
    if search_query:
        data = [d for d in data if search_query.lower() in d['kode_model'].lower()
                or search_query.lower() in d['nama_algoritma'].lower()]

    if not data:
        st.info("Belum ada model yang terhapus")
        return
    
    st.markdown("---")
    st.markdown("<div style='padding-top: 1.2rem;'>", unsafe_allow_html=True)

    if data:
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 3, 2, 2, 2, 2, 1, 1])
        with col1:
            st.write("**Kode Model**")
        with col2:
            st.write("**Algoritma Pembangun**")
        with col3:
            st.write("**Jumlah Fold**")
        with col4:
            st.write("**Akurasi**")
        with col5:
            st.write("**Dibuat**")
        with col6:
            st.write("**Dihapus**")
        with col7:
            st.write("")
        with col8:
            st.write("")

        st.markdown("<div style='padding-top: 1.2rem;'>", unsafe_allow_html=True)

        for row in data:
            kode_model = row["kode_model"]
            nama_algoritma = row["nama_algoritma"]
            tanggal_dibuat = row["tanggal_dibuat"]
            tanggal_dihapus = row["tanggal_dihapus"]
            file_model = row["file_model"]
            fold = row["jumlah_fold"]
            akurasi = row["akurasi"]
            kode_alg = row["kode_algoritma"]
            id_pengguna = row["id_pengguna"]
            nama_pengguna = row["nama_pengguna"]

            # st.markdown("---")
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 3, 2, 2, 2, 2, 1, 1])
            with col1:
                st.write(f"{kode_model}")
            with col2:
                st.write(f"{nama_algoritma}")
            with col3:
                st.write(f"Fold: {fold}")
            with col4:
                st.write(f"{akurasi*100:.2f}%")
            with col5:
                st.write(tanggal_dibuat.strftime("%d %B %Y %H:%M"))
            with col6:
                st.write(tanggal_dihapus.strftime("%d %B %Y %H:%M"))
            with col7:
                if st.button("Pulihkan", key=f"pulihkan_{kode_model}"):
                    st.session_state["pending_restore"] = row
                    st.rerun()
            with col8:
                st.write("")
                
            if st.session_state.get("pending_restore", {}).get("kode_model") == kode_model:
                db = connect_to_db()
                cursor = db.cursor(dictionary=True)

                # 1. Cari model aktif di `models` dengan algoritma & pengguna sama
                cursor.execute("""
                    SELECT * FROM models 
                    WHERE kode_algoritma = %s AND id_pengguna = %s
                """, (kode_alg, id_pengguna))
                existing_model = cursor.fetchone()

                # 2. Jika ada model yang akan digantikan
                if existing_model:
                    colC1, colC2, colC3 = st.columns([8.6, 1, 1.3])
                    with colC1:
                        st.warning(f"""
                            ⚠️ Anda sudah memiliki model **{nama_algoritma}** aktif 
                            ( **`{existing_model['kode_model']}`** ) dengan akurasi {existing_model['akurasi']*100:.2f}%. 
                            Jika Anda melanjutkan, model tersebut akan dipindahkan ke sampah.
                        """)

                    with colC2:
                        tetap_pulihkan = st.button("Tetap Pulihkan", key=f"confirm_pulihkan_{kode_model}")

                    with colC3:
                        batal_pulihkan = st.button("Batal", key=f"batal_pulihkan_{kode_model}")

                    if tetap_pulihkan:
                        # Ambil waktu penghapusan saat ini
                        now = datetime.now()

                        # 1. Pindahkan model aktif (yang akan digantikan) ke arsip_models
                        cursor.execute("""
                            INSERT INTO arsip_models 
                            (kode_model, kode_algoritma, nama_algoritma, file_model, jumlah_fold, akurasi, 
                            tanggal_dibuat, tanggal_dihapus, id_pengguna, nama_pengguna)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """, (
                            existing_model["kode_model"], existing_model["kode_algoritma"],
                            existing_model["nama_algoritma"], existing_model["file_model"],
                            existing_model["jumlah_fold"], existing_model["akurasi"],
                            existing_model["tanggal_dibuat"], now,
                            existing_model["id_pengguna"], existing_model["nama_pengguna"]
                        ))

                        # 2. Hapus model aktif dari tabel models
                        cursor.execute("DELETE FROM models WHERE kode_model = %s", (existing_model["kode_model"],))

                        # 3. Pindahkan model dari arsip ke models (TANPA ubah tanggal_dibuat)
                        cursor.execute("""
                            INSERT INTO models 
                            (kode_model, kode_algoritma, nama_algoritma, file_model, jumlah_fold, akurasi, 
                            tanggal_dibuat, id_pengguna, nama_pengguna)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """, (
                            kode_model, kode_alg, nama_algoritma, file_model, fold, akurasi,
                            tanggal_dibuat, id_pengguna, nama_pengguna
                        ))

                        # 4. Hapus dari arsip_models
                        cursor.execute("DELETE FROM arsip_models WHERE kode_model = %s", (kode_model,))
                        db.commit()
                        st.success("✅ Model berhasil dipulihkan dan menggantikan model sebelumnya.")
                        time.sleep(2)
                        st.rerun()
                    if batal_pulihkan:
                        st.info("❌ Proses pemulihan dibatalkan.")
                        time.sleep(1)
                        st.rerun()
                else:
                    # Jika tidak ada model yang aktif, langsung pindahkan dari arsip ke aktif
                    cursor.execute("""
                        INSERT INTO models 
                        (kode_model, kode_algoritma, nama_algoritma, file_model, jumlah_fold, akurasi, 
                        tanggal_dibuat, id_pengguna, nama_pengguna)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        kode_model, kode_alg, nama_algoritma, file_model, fold, akurasi,
                        tanggal_dibuat, id_pengguna, nama_pengguna
                    ))
                    cursor.execute("DELETE FROM arsip_models WHERE kode_model = %s", (kode_model,))
                    db.commit()
                    cursor.close()
                    db.close()
                    st.success("✅ Model berhasil dipulihkan.")
                    time.sleep(2)
                    st.rerun()
                st.markdown("---")

    st.markdown("---")
    colK,colL = st.columns([1,9])
    with colK:
        if st.button("Kembali"):
            del st.session_state["hal_arsip_model"]
            del st.session_state["info_mining"]
            st.rerun()
