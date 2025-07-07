import streamlit as st
import mysql.connector
import bcrypt
import os
import CSS
import base64
import time
import streamlit.components.v1 as components

def connect_db():
    return mysql.connector.connect(
            host="sql12.freesqldatabase.com",
            user="sql12788718", 
            password="zh24wGnPJN", 
            database="sql12788718" 
    )
def play_success_sound():
    db = connect_db()
    cursor = db.cursor()
    user = st.session_state.user
    user_id = user["id_pengguna"]
    cursor.execute("SELECT nada_notif FROM pengguna WHERE id_pengguna=%s", (user_id,))
    result = cursor.fetchone()
    cursor.close()
    db.close()

    if result and result[0]:
        colj, colk, coll = st.columns([2,2,6])
        with colj:
            if st.button("🔊 Test"):
                encoded_audio = base64.b64encode(result[0]).decode()
                st.markdown(f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
                </audio>
                """, unsafe_allow_html=True)
        with colk:
            if st.button("Ganti"):
                st.session_state.edit_notif = "mode_edit"
                st.rerun()
    else:
        st.info("Kamu belum menambakan nada.")
        if st.button("➕ Tambahkan Nada"):
            st.session_state.edit_notif = "mode_edit"
            st.rerun()
        
def hash_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt)

def check_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode() if isinstance(hashed_password, str) else hashed_password)

def update_user_profile(user_id, nama_pengguna, nama_lengkap, foto_bytes=None, new_password=None):
    db = connect_db()
    cursor = db.cursor()
    if foto_bytes and new_password:
        cursor.execute("""UPDATE pengguna SET nama_pengguna=%s, nama_lengkap=%s, foto_profil=%s, kata_sandi=%s WHERE id_pengguna=%s""",
                       (nama_pengguna, nama_lengkap, foto_bytes, hash_password(new_password), user_id))
    elif foto_bytes:
        cursor.execute("""UPDATE pengguna SET nama_pengguna=%s, nama_lengkap=%s, foto_profil=%s WHERE id_pengguna=%s""",
                       (nama_pengguna, nama_lengkap, foto_bytes, user_id))
    elif new_password:
        cursor.execute("""UPDATE pengguna SET nama_pengguna=%s, nama_lengkap=%s, kata_sandi=%s WHERE id_pengguna=%s""",
                       (nama_pengguna, nama_lengkap, hash_password(new_password), user_id))
    else:
        cursor.execute("""UPDATE pengguna SET nama_pengguna=%s, nama_lengkap=%s WHERE id_pengguna=%s""",
                       (nama_pengguna, nama_lengkap, user_id))
    db.commit()
    cursor.close()
    db.close()

def update_dering(user_id, nada_notif_bytes=None):
    db = connect_db()
    cursor = db.cursor()

    query = "UPDATE pengguna SET nada_notif=%s WHERE id_pengguna=%s"
    params = [nada_notif_bytes]
    params.append(user_id)

    cursor.execute(query, tuple(params))
    db.commit()
    cursor.close()
    db.close()

def get_user_data(user_id):
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM pengguna WHERE id_pengguna=%s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    db.close()
    return user

def main():
    CSS.cssprofil
    st.subheader("Profil Saya")
    st.markdown("<div style='padding-top: 1rem;'>", unsafe_allow_html=True)

    user = st.session_state.user
    user_id = user["id_pengguna"]

    user_data = get_user_data(user_id)
    colA,colB,colC = st.columns([3,3,4])
    with colA:
        st.markdown("## Biodata")
        if "edit_profil" not in st.session_state:
            st.session_state.edit_profil = "mode_normal"
        if st.session_state.edit_profil == "mode_edit":
            with st.container(border=True):
                colQ,colW,colX = st.columns([3.5,0.5,0.01])
                with colQ:
                    st.text_input("ID Pengguna", user_data["id_pengguna"], disabled=True)    
                with colW:
                    st.markdown("<div style='padding-top: 1.8rem;'>", unsafe_allow_html=True)
                    copy_code = f"""
                    <button onclick="navigator.clipboard.writeText('{user_data["id_pengguna"]}'); 
                            var btn = this; btn.innerText='✅'; 
                            setTimeout(()=>{{btn.innerText='📋'}},2000);">
                        📋
                    </button>
                    """
                    components.html(copy_code, height=35)
                nama_pengguna = st.text_input("Nama Pengguna", user_data["nama_pengguna"], disabled=True)
                nama_lengkap = st.text_input("Nama Lengkap", user_data["nama_lengkap"])
                foto_bytes = None
                col1,col2 = st.columns([1,3])
                with col1:
                    st.markdown("<div style='padding-top: 1rem;'>", unsafe_allow_html=True)
                    
                    if 'foto_profil_preview' in st.session_state:
                        encoded_img = base64.b64encode(st.session_state['foto_profil_preview']).decode()
                        st.markdown(
                            f"""
                            <div style="text-align: center;">
                                <img src="data:image/jpeg;base64,{encoded_img}" 
                                    style="width: 100px; height: 100px; border-radius: 50%; border: 2px solid #2657A3; margin-bottom: 1rem;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif user_data["foto_profil"]:
                        encoded_img = base64.b64encode(user_data["foto_profil"]).decode()
                        st.markdown(
                            f"""
                            <div style="text-align: center;">
                                <img src="data:image/jpeg;base64,{encoded_img}" 
                                    style="width: 100px; height: 100px; border-radius: 50%; border: 2px solid #2657A3; margin-bottom: 1rem;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div style="text-align: center;">
                                <img src="https://www.w3schools.com/howto/img_avatar.png" 
                                    style="width: 100px; height: 100px; border-radius: 50%; border: 2px solid #2657A3; margin-bottom: 1rem;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                with col2:
                    foto = st.file_uploader("Ganti Foto Profil", type=["jpg", "png", "jpeg"], key="foto_uploader")

                    # Hanya set preview dan tandai untuk rerun sekali
                    if foto and 'foto_profil_preview' not in st.session_state:
                        st.session_state['foto_profil_preview'] = foto.read()
                        st.session_state['rerun_once'] = True
                        st.rerun()

                    # Jalankan rerun hanya sekali
                    if st.session_state.get('rerun_once'):
                        st.session_state['rerun_once'] = False
                    else:
                        foto_bytes = st.session_state.get('foto_profil_preview', None)

                with st.expander("Ganti Kata Sandi"):
                    old_pass = st.text_input("Masukkan Kata Sandi Lama", type="password")
                    new_pass = st.text_input("Kata Sandi Baru", type="password")
                    confirm_pass = st.text_input("Konfirmasi Kata Sandi Baru", type="password")
                    
                st.markdown("<div style='padding-top: 1rem;'>", unsafe_allow_html=True)
                col6, col7, col8 = st.columns([4,2,5])
                with col6:
                    submitted = st.button("Simpan Perubahan")
                with col7:
                    if st.button("Batal"):
                        st.session_state.edit_profil = "mode_normal"
                        st.rerun()
                if submitted:
                    if 'foto_profil_preview' in st.session_state:
                        del st.session_state['foto_profil_preview']
                    if old_pass:
                        if check_password(old_pass, user_data["kata_sandi"]):
                            if new_pass == confirm_pass and new_pass.strip() != "":
                                update_user_profile(user_id, nama_pengguna, nama_lengkap, foto_bytes, new_pass)
                                st.success("✅ Profil dan kata sandi berhasil diperbarui.")
                                time.sleep(2)
                                st.session_state.edit_profil = "mode_normal"
                                st.rerun()
                            else:
                                st.warning("⚠️ Kata sandi baru tidak cocok atau kosong.")
                        else:
                            st.error("❌ Kata sandi lama tidak sesuai.")
                    else:
                        update_user_profile(user_id, nama_pengguna, nama_lengkap, foto_bytes)
                        st.success("✅ Profil berhasil diperbarui.")
                        time.sleep(2)
                        st.session_state.edit_profil = "mode_normal"
                        st.rerun()

        elif st.session_state.edit_profil == "mode_normal":
            if 'foto_profil_preview' in st.session_state:
                del st.session_state['foto_profil_preview']
            with st.container(border=True):
                colQ,colW,colX = st.columns([3.5,0.5,0.01])
                with colQ:
                    st.text_input("ID Pengguna", user_data["id_pengguna"], disabled=True)    
                with colW:
                    st.markdown("<div style='padding-top: 1.8rem;'>", unsafe_allow_html=True)
                    copy_code = f"""
                    <button onclick="navigator.clipboard.writeText('{user_data["id_pengguna"]}'); 
                            var btn = this; btn.innerText='✅'; 
                            setTimeout(()=>{{btn.innerText='📋'}},2000);">
                        📋
                    </button>
                    """
                    components.html(copy_code, height=35)

                nama_pengguna = st.text_input("Nama Pengguna", user_data["nama_pengguna"], disabled=True)
                nama_lengkap = st.text_input("Nama Lengkap", user_data["nama_lengkap"], disabled=True)
                col1,col2 = st.columns([1,3])
                with col1:
                    st.markdown("<div style='padding-top: 1rem;'>", unsafe_allow_html=True)
                    if user_data["foto_profil"]:
                        encoded_img = base64.b64encode(user_data["foto_profil"]).decode()
                        st.markdown(
                            f"""
                            <div style="text-align: center;">
                                <img src="data:image/jpeg;base64,{encoded_img}" 
                                    style="width: 100px; height: 100px; border-radius: 50%; border: 2px solid #2657A3; margin-bottom: 1rem;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div style="text-align: center;">
                                <img src="https://www.w3schools.com/howto/img_avatar.png" 
                                    style="width: 100px; height: 100px; border-radius: 50%; border: 2px solid #2657A3; margin-bottom: 1rem;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                with col2:
                    foto = st.file_uploader("Ganti Foto Profil", type=["jpg", "png", "jpeg"], disabled=True)
                    foto_bytes = None
                    if foto:
                        foto_bytes = foto.read()

                st.markdown("<div style='padding-top: 1rem;'>", unsafe_allow_html=True)
                if st.button("Edit Profil"):
                    st.session_state.edit_profil = "mode_edit"
                    st.rerun()
    with colB:
        st.markdown("## Nada Notifikasi")
        if "edit_notif" not in st.session_state:
            st.session_state.edit_notif = "mode_normal"
        if st.session_state.edit_notif == "mode_edit":

            with st.container(border=True):

                if "opsi_upload" not in st.session_state:
                    st.session_state.opsi_upload = "pilihan"
                if st.session_state.opsi_upload == "pilihan":
                    colAA, colBB, colCC = st.columns([3,3,3])
                    with colAA:
                        if st.button("🔔 Default Sistem"):
                            st.session_state.opsi_upload = "default"
                            st.rerun()
                    with colBB:
                        if st.button("📤 Upload Nada"):
                            st.session_state.opsi_upload = "unggah"
                            st.rerun()
                    st.markdown("<div style='padding-top: 2rem;'>", unsafe_allow_html=True)
                    if st.button("Kembali"):
                        st.session_state.edit_notif = "mode_normal"
                        st.rerun()

                elif st.session_state.opsi_upload == "default":
                    st.markdown("##### Pilih Nada Default dari Sistem")
                    db = connect_db()
                    cursor = db.cursor(dictionary=True)
                    cursor.execute("SELECT kode_notif, file_notif FROM notifikasi ORDER BY tanggal_dibuat DESC")
                    notifs = cursor.fetchall()
                    cursor.close()
                    db.close()

                    if notifs:
                        for notif in notifs:
                            kode_notif = notif['kode_notif']
                            file_bytes = notif['file_notif']

                            colA, colB, colC = st.columns([4.5, 1, 1.5])
                            with colA:
                                st.markdown(f"**Nada Kode: `{kode_notif}`**")
                            with colB:
                                ButtonTest = st.button("Test", key=f"test_{kode_notif}")
                            with colC:
                                if st.button("Terapkan", key=f"apply_{kode_notif}"):
                                    db = connect_db()
                                    cursor = db.cursor()
                                    cursor.execute(
                                        "UPDATE pengguna SET nada_notif = %s WHERE id_pengguna = %s",
                                        (file_bytes, user_id)
                                    )
                                    db.commit()
                                    cursor.close()
                                    db.close()
                                    st.success(f"✅ Nada `{kode_notif}` berhasil diterapkan.")
                                    st.session_state.edit_notif = "mode_normal"
                                    del st.session_state['opsi_upload']
                                    time.sleep(1)
                                    st.rerun()
                            if ButtonTest:
                                encoded_audio = base64.b64encode(file_bytes).decode()
                                st.markdown(f"""
                                <div id="audio-container">
                                    <audio id="notif-audio" autoplay onended="document.getElementById('audio-container').style.display='none'">
                                        <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
                                        Browser Anda tidak mendukung pemutar audio.
                                    </audio>
                                </div>
                                """, unsafe_allow_html=True)
                        if st.button("Batal"):
                            st.session_state.opsi_upload = "pilihan"
                            st.rerun()
                    else:
                        st.info("📭 Belum ada nada sistem tersedia.")
                
                elif st.session_state.opsi_upload == "unggah":
                    mp3 = st.file_uploader("Unggah Nada Notifikasi", type=["mp3"], key="upload_mp3")
                    if mp3:
                        mp3_bytes = mp3.read()
                        st.session_state['nada_notif_preview'] = mp3_bytes
                        col_a, col_b = st.columns([6,4])
                        with col_a:
                            st.success("Nada berhasil diunggah,")
                        with col_b:
                            Dengarkan = st.button("Dengarkan")
                        if Dengarkan:
                            encoded_audio = base64.b64encode(mp3_bytes).decode()
                            st.markdown(f"""
                            <audio controls autoplay>
                                <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
                                Browser Anda tidak mendukung tag audio.
                            </audio>
                            """, unsafe_allow_html=True)

                        st.markdown("<div style='padding-top: 1rem;'>", unsafe_allow_html=True)
                        colD, colE, colF = st.columns([4,2,5])
                        with colD:
                            simpan_nada = st.button("Simpan Perubahan")
                        with colE:
                            if st.button("Batal"):
                                st.session_state.opsi_upload = "pilihan"
                                st.rerun()
                        if simpan_nada:
                            if 'nada_preview' in st.session_state:
                                del st.session_state['nada_preview']
                            nada_notif_bytes = st.session_state.get('nada_notif_preview', None)
                            if nada_notif_bytes:
                                update_dering(user_id, nada_notif_bytes)
                                st.success("✅ Dering berhasil diperbarui.")
                                time.sleep(2)
                                st.session_state.edit_notif = "mode_normal"
                                del st.session_state['opsi_upload']
                                st.rerun()
                            else:
                                st.warning("waduh gagal menyimpan nada")
                    if mp3 is None:
                        if st.button("Batal "):
                            st.session_state.opsi_upload = "pilihan"
                            st.rerun()

        elif st.session_state.edit_notif == "mode_normal":
            if 'nada_preview' in st.session_state:
                del st.session_state['nada_preview']

            with st.container(border=True):
                    play_success_sound()



