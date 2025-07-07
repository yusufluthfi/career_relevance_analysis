import streamlit as st
import mysql.connector
import time
import streamlit.components.v1 as components
import CSS


def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="data_mining_system"
    )

def main():

    st.subheader("Kelola Pengguna")
    st.markdown("Berikut data pengguna sistem prediksi karier")

    db = connect_to_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM pengguna")
    data = cursor.fetchall()
    cursor.close()
    db.close()

    col5, col6, col7 = st.columns([6, 5, 3])
    with col5:
        search_query = st.text_input("🔍 Cari Nama Pengguna", placeholder="Cari nama pengguna...", label_visibility="hidden")
    with col7 :
        st.markdown("<div style='padding-top: 1.8rem;'>", unsafe_allow_html=True)
        if st.button("Buat User Baru"):
            st.session_state.halaman = "form_tambah_pengguna"
            st.rerun()

    st.markdown("---")

    if search_query:
        data = [user for user in data if search_query.lower() in nama_pengguna.lower()]

    if "edit_kode" not in st.session_state:
        st.session_state.edit_kode = None

    if "edit_confirm" not in st.session_state:
        st.session_state.edit_confirm = None
        
    if "delete_confirm" not in st.session_state:
        st.session_state.delete_confirm = None  # Menyimpan ID pengguna yang ingin dihapus

    if "delete_success" not in st.session_state:
        st.session_state.delete_success = None  # Menyimpan ID pengguna yang berhasil dihapus


    if data:
        # Header
        col_h1, col_h2, col_h3, col_h4, col_h5, col_h6, col_h7 = st.columns([1.5, 2, 1.5, 1.5, 0.5, 0.5, 4])
        with col_h1:
            st.write("**ID**")
        with col_h2:
            st.write("**Nama**")
        with col_h3:
            st.write("**Username**")
        with col_h4:
            st.write("**Hak Akses**")
        with col_h5:
            st.write("")
        with col_h6:
            st.write("")
        st.markdown("")

        for user in data:

            id_pengguna = user["id_pengguna"]
            nama_lengkap = user["nama_lengkap"]
            nama_pengguna = user["nama_pengguna"]
            hak_akses = user["hak_akses"]
            
            # mode edit
            if st.session_state.edit_kode == id_pengguna:

                col1, col2, col3, col4,col5,col6, col7 = st.columns([1.5, 2, 1.5, 1.5, 0.5, 0.5,4])
                with col1:
                    components.html(f"""
                        <button onclick="navigator.clipboard.writeText('{id_pengguna}'); alert('ID Pengguna: {id_pengguna} berhasil disalin!')" style="
                            padding: 6px 12px;
                            background-color: #4CAF50;
                            color: white;
                            border: none;
                            border-radius: 5px;
                            cursor: pointer;
                        ">
                            Lihat
                        </button>
                    """, height=40)
                with col2:
                    st.write(nama_lengkap)
                with col3:
                    st.write(nama_pengguna)
                with col4:
                    new_access = st.selectbox(
                        f"", 
                        ["user", "admin"], 
                        index=["user", "admin"].index(user["hak_akses"]),
                        key=f"select_{id_pengguna}",label_visibility="collapsed"
                    )
                with col5:
                    if st.button("💾"):
                        if new_access.strip() == hak_akses.strip():
                            st.info("Tidak ada perubahan data.")
                            st.session_state.edit_kode = None
                            st.rerun()
                        else:
                            st.session_state.edit_confirm = (id_pengguna, new_access)
                            st.rerun()
                with col6:
                    if st.button("❌"):
                        st.session_state.edit_kode = None
                        st.rerun()
            # mode normal
            else:
                col1, col2, col3, col4,col5,col6, col7 = st.columns([1.5, 2, 1.5, 1.5, 0.5, 0.5, 4])
                with col1:
                    components.html(f"""
                        <button onclick="navigator.clipboard.writeText('{id_pengguna}'); alert('ID Pengguna: {id_pengguna} berhasil disalin!')" style="
                            padding: 6px 12px;
                            background-color: #4CAF50;
                            color: white;
                            border: none;
                            border-radius: 5px;
                            cursor: pointer;
                        ">
                            Lihat
                        </button>
                    """, height=40)
                with col2:
                    st.write(nama_lengkap)
                with col3:
                    st.write(nama_pengguna)
                with col4:
                    st.write(hak_akses)
                with col5:
                    if st.button("✏️", key=f"edit_{id_pengguna}"):
                        st.session_state.edit_kode = id_pengguna
                        st.rerun()
                with col6:
                    if st.button("🗑️", key=f"delete_{id_pengguna}"):
                        st.session_state.delete_confirm = user  

        # Pop-up konfirmasi penghapusan
        if st.session_state.delete_confirm:
            user = st.session_state.delete_confirm
            st.warning(f"⚠️ Apakah Anda yakin ingin menghapus akun **{nama_pengguna}**?")
            colC1, colC2 = st.columns(2)
            with colC1:
                if st.button("✅ Ya, Hapus"):
                    delete_user(user["id_pengguna"])
                    st.session_state.delete_success = nama_pengguna
                    st.session_state.delete_confirm = None
                    st.rerun()
            with colC2:
                if st.button("❌ Batal"):
                    st.session_state.delete_confirm = None
                    st.rerun()

        # Notifikasi sukses penghapusan
        if st.session_state.delete_success:
            st.success(f"Akun **{st.session_state.delete_success}** berhasil dihapus.")
            time.sleep(2)
            st.session_state.delete_success = None
            st.rerun()

        # Pop-up konfirmasi simpan edit
        if st.session_state.edit_confirm:
            user_id, akses_baru = st.session_state.edit_confirm
            st.warning(f"⚠️ Yakin ingin mengubah hak akses **{nama_pengguna}** menjadi: **{akses_baru}** ?")
            colS1, colS2, colS3 = st.columns([1.5,1,8])
            with colS1:
                if st.button("Ya, Simpan Perubahan"):
                    update_access(user_id, akses_baru)
                    st.success("✅ Hak akses berhasil diperbarui.")
                    st.session_state.edit_kode = None
                    st.session_state.edit_confirm = None
                    time.sleep(1)
                    st.rerun()
            with colS2:
                if st.button("Batal Edit"):
                    st.session_state.edit_confirm = None
                    st.rerun()

        # Inisialisasi nilai default di session_state
    if "new_access" not in st.session_state:
        st.session_state.new_access = ""
    if "new_access_pengguna" not in st.session_state:
        st.session_state.new_access_pengguna = ""
    if "new_kata_sandi" not in st.session_state:
        st.session_state.new_kata_sandi = ""
    if "new_hak_akses" not in st.session_state:
        st.session_state.new_hak_akses = "user"

    st.markdown("---")
    st.markdown("")
    # Tombol kembali
    # if st.button("↩ Kembali"):
    #     st.rerun()


def update_access(user_id, akses_baru):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("UPDATE pengguna SET hak_akses = %s WHERE id_pengguna = %s", (akses_baru, user_id))
    db.commit()
    cursor.close()
    db.close()

def delete_user(user_id):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM pengguna WHERE id_pengguna = %s", (user_id,))
    db.commit()
    cursor.close()
    db.close()

def add_user(nama,username, password, hak_akses):
    import bcrypt, uuid
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    user_id = str(uuid.uuid4())

    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("INSERT INTO pengguna (id_pengguna, nama_lengkap, nama_pengguna, kata_sandi, hak_akses) VALUES (%s, %s, %s, %s, %s)", (user_id, nama, username, hashed, hak_akses))
    db.commit()
    cursor.close()
    db.close()
