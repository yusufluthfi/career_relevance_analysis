import streamlit as st
import mysql.connector
import pandas as pd
import base64
import time

def connect_to_db():
    return mysql.connector.connect(
            host="sql12.freesqldatabase.com",
            user="sql12788718", 
            password="zh24wGnPJN", 
            database="sql12788718" 
    )

def get_model_detail(kode_model):
    db = connect_to_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM models WHERE kode_model = %s", (kode_model,))
    result = cursor.fetchone()
    cursor.close()
    db.close()
    return result

def update_model_detail(kode_model, nama_algoritma, jumlah_fold):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("""
        UPDATE models 
        SET nama_algoritma = %s, jumlah_fold = %s
        WHERE kode_model = %s
    """, (nama_algoritma, jumlah_fold, kode_model))
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
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Unduh Model</a>'
    return href

def main():
    kode_model = st.session_state.kode_model_terpilih
    model = get_model_detail(kode_model)
    
    st.subheader(f"📄 Detail Model {kode_model}")

    if "kode_model_terpilih" not in st.session_state:
        st.warning("⚠️ Tidak ada model yang dipilih.")
        return

    if not model:
        st.error(f"Model dengan kode {kode_model} tidak ditemukan.")
        return

    # State edit/hapus
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "delete_confirm" not in st.session_state:
        st.session_state.delete_confirm = False

    if not st.session_state.edit_mode:
        # Tampilkan detail
        with st.expander("📊 Informasi Model", expanded=True):
            st.write(f"**Kode Model:** {model['kode_model']}")
            st.write(f"**Kode Algoritma:** {model['kode_algoritma']}")
            st.write(f"**Nama Algoritma:** {model['nama_algoritma']}")
            st.write(f"**Jumlah Fold:** {model['jumlah_fold']}")
            st.write(f"**Akurasi:** {model['akurasi'] * 100:.2f}%")
            st.write(f"**Tanggal Dibuat:** {pd.to_datetime(model['tanggal_dibuat']).strftime('%d %B %Y %H:%M')}")
            st.write(f"**ID Pengguna:** {model['id_pengguna']}")
            st.write(f"**Nama Pengguna:** {model['nama_pengguna']}")
            st.markdown(download_button(model["file_model"], f"{model['kode_model']}.pkl"), unsafe_allow_html=True)

        # Tombol aksi
        col1, col2, col3 = st.columns([0.5,0.8,9])
        with col1:
            if st.button("Edit"):
                st.session_state.edit_mode = True
                st.rerun()
        with col2:
            if st.button("Hapus Data"):
                st.session_state.delete_confirm = True
                st.rerun()
        with col3:
            if st.button("Kembali"):
                del st.session_state.kode_model_terpilih
                st.session_state.hal_kelola_model = "kelola_model"
                st.rerun()
    else:
        # Mode edit
        with st.form("form_edit_model"):
            nama_algoritma = st.text_input("Nama Algoritma", value=model['nama_algoritma'])
            jumlah_fold = st.number_input("Jumlah Fold", value=model['jumlah_fold'], min_value=2, step=1)
            colA, colB, colC = st.columns([1,0.5,8])
            with colA:
                simpan = st.form_submit_button("Simpan Perubahan")
            with colB:
                batal = st.form_submit_button("Batal")
            if simpan:
                update_model_detail(kode_model, nama_algoritma, jumlah_fold)
                st.success("✅ Model berhasil diperbarui.")
                st.session_state.edit_mode = False
                time.sleep(1)
                st.rerun()
            elif batal:
                st.session_state.edit_mode = False
                st.rerun()

    # Konfirmasi hapus
    if st.session_state.delete_confirm:
        st.warning(f"⚠️ Yakin ingin menghapus model dengan kode **{kode_model}**?")
        colC1, colC2 = st.columns(2)
        with colC1:
            if st.button("✅ Ya, Hapus Sekarang"):
                delete_model(kode_model)
                st.success("🗑️ Model berhasil dihapus.")
                del st.session_state.delete_confirm
                del st.session_state.kode_model_terpilih
                time.sleep(1)
                st.rerun()
        with colC2:
            if st.button("❌ Batal Hapus"):
                st.session_state.delete_confirm = False
                st.rerun()
