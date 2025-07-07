import streamlit as st
import mysql.connector
import pandas as pd
import base64
import time
import uuid
from datetime import datetime

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="data_mining_system"
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
    current_user = st.session_state.user["nama_pengguna"]
    cursor.execute("""
        SELECT kode_model, kode_algoritma, nama_algoritma, file_model, jumlah_fold, akurasi, 
            tanggal_dibuat, id_pengguna, nama_pengguna
        FROM models
        WHERE nama_pengguna != %s
        ORDER BY tanggal_dibuat DESC
    """, (current_user,))
    data = cursor.fetchall()
    cursor.close()
    db.close()

    col_1, col_2, col_3 = st.columns([7, 1,1.2])
    with col_1:
        st.subheader("Import Model Machine Learning")
        st.markdown("Berikut adalah daftar model bisa anda import")
    with col_2:
        st.caption("Sort by:")
        sort_option = st.selectbox(
            "Sort by",
            options=["Terbaru", "Terlama", "Kode", "Algoritma", "Pembuat"],
            index=0,
            label_visibility="collapsed"
        )

    # Sort By
    if not data:
        st.info("😅 Upss... kamu belum pernah melakukan prediksi nih.")
        return

    # Konversi list of dicts ke DataFrame
    data_model = pd.DataFrame(data)

    # Konversi kolom tanggal
    data_model['tanggal_dibuat'] = pd.to_datetime(data_model['tanggal_dibuat'], errors='coerce')

    # Terapkan urutan sesuai pilihan dropdown
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

    # Kembalikan ke list of dicts
    data = data_model.to_dict(orient='records')

    # Pencarian
    col1, col2 = st.columns([7, 3])
    with col1:
        search_query = st.text_input("Pencarian", placeholder="Cari Kode atau Nama Algoritma", label_visibility="hidden")
        try:
            jumlah_model = len(data)
            st.caption(f"**{jumlah_model}** model tersedia")
        except:
            st.caption("⚠️ Gagal membaca data.")

    st.markdown("---")

    # Filter
    if search_query:
        data = [
            row for row in data if
            search_query.lower() in row["kode_model"].lower() or
            search_query.lower() in row["nama_algoritma"].lower() or
            search_query.lower() in row["nama_pengguna"].lower()
        ]

    if data:
        col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 1, 1.5, 1, 1.5, 1, 6])
        with col2:
            st.write("**Algoritma Pembangun**")
        with col3:
            st.write("**Jumlah Fold**")
        with col4:
            st.write("**Versi**")
        with col5:
            st.write("**Akurasi**")
        with col6:
            st.write("**Tanggal Dibuat**")
        with col7:
            st.write("")
        with col8:
            st.write("")

        st.markdown("<div style='padding-top: 1.2rem;'>", unsafe_allow_html=True)
        for row in data:
            kode_model = row["kode_model"]
            nama_algoritma = row["nama_algoritma"]
            tanggal = pd.to_datetime(row["tanggal_dibuat"]).strftime("%d %B %Y %H:%M")
            penguji = row["nama_pengguna"]
            akurasi = f"{row['akurasi'] * 100:.2f}%"
            fold = row["jumlah_fold"]
            kode_alg = row["kode_algoritma"]
            file_model = row["file_model"]

            col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 1, 1.5, 1, 1.5, 1, 6])
            with col2:
                st.write(nama_algoritma)
            with col3:
                st.write(f"{fold} Fold")
            with col4:
                st.markdown(download_button(file_model, f"{kode_model}.pkl"), unsafe_allow_html=True)
            with col5:
                st.write(akurasi)
            with col6:
                st.write(tanggal)
            with col7:
                if st.button("Import", key=f"import_{kode_model}"):
                    # Simpan semua data yang diperlukan ke session state
                    st.session_state["pending_import"] = {
                        "target_kode_model": kode_model,  # digunakan untuk mencocokkan posisi baris
                        "kode_model_baru": generate_kode_model(),
                        "kode_model_lama": None,
                        "data": {
                            "kode_alg": kode_alg,
                            "nama_algoritma": nama_algoritma,
                            "file_model": file_model,
                            "fold": fold,
                            "akurasi": row['akurasi'],
                            "tanggal_dibuat": datetime.now(),
                            "nama_pengguna": st.session_state.user["nama_pengguna"],
                            "id_pengguna": st.session_state.user["id_pengguna"]
                        }
                    }

                    # Cek model duplikat
                    db = connect_to_db()
                    cursor = db.cursor(dictionary=True)
                    cursor.execute("""
                        SELECT kode_model FROM models 
                        WHERE nama_algoritma = %s AND nama_pengguna = %s
                    """, (nama_algoritma, st.session_state.user["nama_pengguna"]))
                    existing_model = cursor.fetchone()
                    cursor.close()
                    db.close()

                    if existing_model:
                        st.session_state["pending_import"]["kode_model_lama"] = existing_model["kode_model"]
                    else:
                        st.session_state["pending_import"]["confirmed"] = True  # langsung lanjut jika tidak duplikat

                    st.rerun()
            with col8:
                pending = st.session_state.get("pending_import")

                # Tampilkan hanya jika pending dan cocok dengan baris ini
                if pending and pending["target_kode_model"] == kode_model and "confirmed" not in pending:
                    st.warning(
                        f"⚠️ Ini akan menggantikan model `{pending['kode_model_lama']}` "
                        f"karena Anda tidak diizinkan menggunakan algoritma ganda."
                    )
                    colC1, colC2, colC3 = st.columns([1.5,1.5,6])
                    with colC1:
                        if st.button("Tetap Import", key=f"confirm_import_{kode_model}"):
                            db = connect_to_db()
                            cursor = db.cursor()

                            if pending["kode_model_lama"]:
                                cursor.execute("DELETE FROM models WHERE kode_model = %s", (pending["kode_model_lama"],))

                            d = pending["data"]
                            insert_query = """
                                INSERT INTO models (
                                    kode_model, kode_algoritma, nama_algoritma, file_model, 
                                    jumlah_fold, akurasi, tanggal_dibuat, id_pengguna, nama_pengguna
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            cursor.execute(insert_query, (
                                pending["kode_model_baru"],
                                d["kode_alg"],
                                d["nama_algoritma"],
                                d["file_model"],
                                d["fold"],
                                d["akurasi"],
                                d["tanggal_dibuat"],
                                d["id_pengguna"],
                                d["nama_pengguna"]
                            ))
                            db.commit()
                            cursor.close()
                            db.close()

                            st.success(f"✅ Model berhasil diimpor sebagai `{pending['kode_model_baru']}`")
                            time.sleep(2)
                            del st.session_state["pending_import"]
                            st.rerun()

                    with colC2:
                        if st.button("Batal Import", key=f"cancel_import_{kode_model}"):
                            del st.session_state["pending_import"]
                            st.rerun()

                # Jika sudah dikonfirmasi sebelumnya dan ini baris yang sesuai, lanjut simpan
                elif pending and pending["target_kode_model"] == kode_model and pending.get("confirmed"):
                    db = connect_to_db()
                    cursor = db.cursor()
                    d = pending["data"]

                    insert_query = """
                        INSERT INTO models (
                            kode_model, kode_algoritma, nama_algoritma, file_model, 
                            jumlah_fold, akurasi, tanggal_dibuat, id_pengguna, nama_pengguna
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        pending["kode_model_baru"],
                        d["kode_alg"],
                        d["nama_algoritma"],
                        d["file_model"],
                        d["fold"],
                        d["akurasi"],
                        d["tanggal_dibuat"],
                        d["id_pengguna"],
                        d["nama_pengguna"]
                    ))
                    db.commit()
                    cursor.close()
                    db.close()
                    st.success(f"✅ Model berhasil diimpor sebagai `{pending['kode_model_baru']}`")
                    del st.session_state["pending_import"]
                    st.rerun()

    else:
        st.info("📭 Belum ada model atau hasil pencarian tidak ditemukan.")

    st.markdown("---")
