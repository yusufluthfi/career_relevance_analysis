import streamlit as st
import mysql.connector
from datetime import datetime
import time
import CSS

def main():
    CSS.csstambahpengguna()
    st.subheader("Tambah Algoritma Baru")

    with st.form("form_tambah_algoritma"):
        kode_algoritma = st.text_input("Kode Algoritma", max_chars=10, placeholder="Contoh: DT, NB, RF")
        nama_algoritma = st.text_input("Nama Algoritma", placeholder="Contoh: Decision Tree, Naive Bayes")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        submit = st.form_submit_button("Simpan")

        if submit:
            if not kode_algoritma or not nama_algoritma:
                st.warning("Semua kolom wajib diisi.")
                return

            try:
                conn = mysql.connector.connect(
                    host="sql12.freesqldatabase.com",
                    user="sql12788718", 
                    password="zh24wGnPJN", 
                    database="sql12788718" 
                )
                cursor = conn.cursor()

                # Cek apakah kode_algoritma sudah ada
                cursor.execute("SELECT * FROM algoritma WHERE kode_algoritma = %s", (kode_algoritma,))
                if cursor.fetchone():
                    st.error("Kode algoritma sudah terdaftar.")
                else:
                    insert_query = """
                        INSERT INTO algoritma (kode_algoritma, nama_algoritma, tanggal_dibuat)
                        VALUES (%s, %s, %s)
                    """
                    cursor.execute(insert_query, (kode_algoritma, nama_algoritma, now))
                    conn.commit()
                    st.success("✅ Algoritma berhasil ditambahkan.")
                    time.sleep(2)
                    st.session_state.hal_kelola_algoritma = "kelola_algoritma"
                    st.rerun()


            except mysql.connector.Error as e:
                st.error(f"❌ Gagal menyimpan ke database: {e}")
            finally:
                cursor.close()
                conn.close()

    # Tombol kembali
    if st.button("↩ Kembali"):
        st.session_state.hal_kelola_algoritma = "kelola_algoritma"
        st.rerun()
