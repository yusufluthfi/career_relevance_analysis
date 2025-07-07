import streamlit as st
import mysql.connector
import pandas as pd
import base64
import time
from datetime import datetime

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="data_mining_system"
    )

def update_model(kode_model, nama_baru):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("UPDATE models SET nama_algoritma = %s WHERE kode_model = %s", (nama_baru, kode_model))
    db.commit()
    cursor.close()
    db.close()

def delete_model(kode_model):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM models WHERE kode_model = %s", (kode_model,))
    db.commit()
    cursor.close()
    db.close()

def download_button(file_blob, filename):
    b64 = base64.b64encode(file_blob).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Unduh</a>'
    return href

def main(sort_option="Terbaru", search_query=""):

    db = connect_to_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT kode_model, kode_algoritma, nama_algoritma, file_model, jumlah_fold, akurasi, 
               tanggal_dibuat, id_pengguna, nama_pengguna
        FROM models
        ORDER BY tanggal_dibuat DESC
    """)
    data = cursor.fetchall()
    cursor.close()
    db.close()

    col_1, col_2, col_3 = st.columns([7, 1,1.2])
    with col_1:
        st.subheader("Kelola Model Machine Learning")
        st.markdown("Berikut adalah daftar model yang tersimpan dalam sistem.")
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
        search_query = st.text_input("Pencarian", placeholder="Cari Kode, Nama Algoritma, atau Pengguna", label_visibility="hidden")
        try:
            jumlah_model = len(data)
            st.caption(f"Total model: **{jumlah_model}**")
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

    # State untuk aksi
    if "edit_kode_model" not in st.session_state:
        st.session_state.edit_kode_model = None
    if "delete_model_confirm" not in st.session_state:
        st.session_state.delete_model_confirm = None
    if "edit_model_confirm" not in st.session_state:
        st.session_state.edit_model_confirm = None

    if data:
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([2, 3, 2, 2, 2, 2, 1, 1, 1, 1])
        with col1:
            st.write("**Kode Model**")
        with col2:
            st.write("**Algoritma Pembangun**")
        with col3:
            st.write("**File Model**")
        with col4:
            st.write("**Jumlah Fold**")
        with col5:
            st.write("**Akurasi**")
        with col6:
            st.write("**Tanggal Dibuat**")
        with col7:
            st.write("**Pembuat**")
        with col8:
            st.write("")
        with col9:
            st.write("")
        with col10:
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

            if st.session_state.edit_kode_model == kode_model:
                col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([2, 3, 2, 2, 2, 2, 1, 1, 1])
                with col1:
                    st.markdown(f"<span style='color: orange; font-weight: bold;'>{kode_model}</span>", unsafe_allow_html=True)
                with col2:
                    new_nama = st.text_input("", value=nama_algoritma, key=f"edit_nama_{kode_model}", label_visibility="collapsed")
                with col3:
                    st.markdown(download_button(file_model, f"{kode_model}.pkl"), unsafe_allow_html=True)
                with col4:
                    st.write(f"{fold} Fold")
                with col5:
                    st.write(akurasi)
                with col6:
                    st.write(tanggal)
                with col7:
                    if st.button("💾", key=f"simpan_{kode_model}"):
                        if new_nama.strip() == nama_algoritma.strip():
                            st.info("Tidak ada perubahan.")
                            st.session_state.edit_kode_model = None
                            st.rerun()
                        else:
                            st.session_state.edit_model_confirm = (kode_model, new_nama)
                            st.rerun()
                with col8:
                    if st.button("❌", key=f"batal_edit_{kode_model}"):
                        st.session_state.edit_kode_model = None
                        st.rerun()
                with col9:
                    st.write(penguji)
            else:
                col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([2, 3, 2, 2, 2, 2, 1, 1, 1, 1])
                with col1:
                    st.write(kode_model)
                with col2:
                    st.write(nama_algoritma)
                with col3:
                    st.markdown(download_button(file_model, f"{kode_model}.pkl"), unsafe_allow_html=True)
                with col4:
                    st.write(f"{fold} Fold")
                with col5:
                    st.write(akurasi)
                with col6:
                    st.write(tanggal)
                with col7:
                    st.write(penguji)
                with col8:
                    if st.button("Detail", key=f"detail_{kode_model}"):
                        st.session_state.kode_model_terpilih = kode_model
                        st.session_state.hal_kelola_model = "detail_model"
                        st.rerun()
                # with col9:
                #     if st.button("✏️", key=f"edit_{kode_model}"):
                #         st.session_state.edit_kode_model = kode_model
                #         st.rerun()
                # with col10:
                #     if st.button("🗑️", key=f"hapus_{kode_model}"):
                #         st.session_state.delete_model_confirm = kode_model
                #         st.rerun()
    else:
        st.info("📭 Belum ada model atau hasil pencarian tidak ditemukan.")

    st.markdown("---")

    # Konfirmasi hapus
    if st.session_state.delete_model_confirm:
        kode = st.session_state.delete_model_confirm
        st.warning(f"⚠️ Yakin ingin menghapus model dengan kode **{kode}**?")
        colC1, colC2 = st.columns(2)
        with colC1:
            if st.button("✅ Ya, Hapus"):
                delete_model(kode)
                st.success("🗑️ Model berhasil dihapus.")
                st.session_state.delete_model_confirm = None
                time.sleep(1)
                st.rerun()
        with colC2:
            if st.button("❌ Batal"):
                st.session_state.delete_model_confirm = None
                st.rerun()

    # Konfirmasi simpan edit
    if st.session_state.edit_model_confirm:
        kode, nama_baru = st.session_state.edit_model_confirm
        st.warning(f"⚠️ Yakin ingin mengubah nama algoritma model **{kode}** menjadi:\n\n➡️ **{nama_baru}** ?")
        colS1, colS2 = st.columns(2)
        with colS1:
            if st.button("✅ Simpan Perubahan"):
                update_model(kode, nama_baru)
                st.success("✅ Nama algoritma model berhasil diperbarui.")
                st.session_state.edit_kode_model = None
                st.session_state.edit_model_confirm = None
                time.sleep(1)
                st.rerun()
        with colS2:
            if st.button("❌ Batal"):
                st.session_state.edit_model_confirm = None
                st.rerun()
