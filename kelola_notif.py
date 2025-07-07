import streamlit as st
import mysql.connector
import base64
import uuid
from datetime import datetime
import time
import CSS

def connect_to_db():
    return mysql.connector.connect(
            host="sql12.freesqldatabase.com",
            user="sql12788718", 
            password="zh24wGnPJN", 
            database="sql12788718" 
    )

def generate_kode_notif(prefix="NTF"):
    return f"{prefix}.{uuid.uuid4().hex[:6].upper()}"

def audio_player(file_bytes):
    encoded_audio = base64.b64encode(file_bytes).decode()
    st.markdown(f"""
    <div id="audio-container-temp">
        <audio id="audio-preview" controls autoplay onended="document.getElementById('audio-container-temp').remove();">
            <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
            Browser Anda tidak mendukung tag audio.
        </audio>
    </div>
    """, unsafe_allow_html=True)

def main():
    CSS.csstambahpengguna
    st.subheader("Kelola Notifikasi")

    # Uploader dan preview
    with st.container(border=True):
        st.markdown("##### Upload Nada Notifikasi Baru")

        mp3 = st.file_uploader("Pilih File MP3", type=["mp3"], key="mp3_upload")
        if mp3:
            mp3_bytes = mp3.read()
            st.session_state["preview_notif"] = mp3_bytes
            st.success("Nada berhasil dipilih. Silakan preview dan unggah.")

            col1, col2, col3, col4 = st.columns([0.7, 0.6, 0.5, 9])
            with col1:
                if st.button("Test Nada"):
                    audio_player(mp3_bytes)
            with col2:
                if st.button("Unggah"):
                    db = connect_to_db()
                    cursor = db.cursor()
                    kode = generate_kode_notif()
                    cursor.execute(
                        "INSERT INTO notifikasi (kode_notif, file_notif, tanggal_dibuat) VALUES (%s, %s, %s)",
                        (kode, mp3_bytes, datetime.now())
                    )
                    db.commit()
                    cursor.close()
                    db.close()
                    st.success("✅ Nada berhasil diunggah.")
                    del st.session_state["preview_notif"]
                    time.sleep(2)
                    st.rerun()
            with col3:
                bataltambah=st.button("Batal")
            with col4:
                if bataltambah:
                    del st.session_state["preview_notif"]
                    mp3 = None
                    st.warning("❌ Upload dibatalkan.")
                    time.sleep(1)
                    st.rerun()

    # Menampilkan daftar nada
    st.markdown("---")
    st.markdown("##### Daftar Nada Notifikasi yang Tersimpan")
    db = connect_to_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM notifikasi ORDER BY tanggal_dibuat DESC")
    data = cursor.fetchall()
    cursor.close()
    db.close()

    if not data:
        st.info("📭 Belum ada nada yang diunggah.")
        return

    colA, colB = st.columns([3, 10.5])
    with colA:
        for row in data:
            with st.container(border=True):
                st.write(f"**Kode:** `{row['kode_notif']}` | **Tanggal:** `{row['tanggal_dibuat'].strftime('%d %B %Y %H:%M')}`")
                col1, col2, col3 = st.columns([2, 2.5, 6])
                with col1:
                    if st.button("Test", key=f"test_{row['kode_notif']}"):
                        audio_player(row['file_notif'])
                with col2:
                    if st.button("Hapus", key=f"hapus_{row['kode_notif']}"):
                        st.session_state['confirm_delete'] = row['kode_notif']
                        st.rerun()
                if st.session_state.get('confirm_delete') == row['kode_notif']:
                    st.warning(f"⚠️ Apakah Anda yakin ingin menghapus nada `{row['kode_notif']}`?")
                    col_confirm1, col_confirm2 = st.columns([1, 1])
                    with col_confirm1:
                        if st.button("Ya, Hapus", key=f"confirm_delete_{row['kode_notif']}"):
                            db = connect_to_db()
                            cursor = db.cursor()
                            cursor.execute("DELETE FROM notifikasi WHERE kode_notif = %s", (row['kode_notif'],))
                            db.commit()
                            cursor.close()
                            db.close()
                            st.success("✅ Nada berhasil dihapus.")
                            del st.session_state['confirm_delete']
                            time.sleep(1)
                            st.rerun()
                    with col_confirm2:
                        if st.button("Batal", key=f"cancel_delete_{row['kode_notif']}"):
                            del st.session_state['confirm_delete']
                            st.rerun()
