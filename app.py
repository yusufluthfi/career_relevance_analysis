import os
import streamlit as st
import pandas as pd
import time
import joblib
import uuid
import login
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import io
import mysql.connector
import kelola_pengguna
import form_tambah_pengguna 
import dashboard
import base64
import CSS
import tambah_algoritma
import kelola_algoritma
import kelola_model
import detail_model
import profil
import kelola_notif
import import_model
import arsip_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from decision_tree import run_decision_tree
from random_forest import run_random_forest
from LR import run_logistic_regression
from svm import run_svm
from XGBoost import run_xgboost
from naive_bayes import run_naive_bayes
from AdaBoost import run_adaboost
from Cat_Boost import run_catboost
from KNN import run_knn
from Voting_Classifier import run_voting_classifier
from AdaBoost_XGBoost import run_AdaBoost_XGBoost
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from streamlit_option_menu import option_menu
from PIL import Image


def main() :

    def connect_to_db():
        return mysql.connector.connect(
            host="localhost",
            user="root", 
            password="", 
            database="data_mining_system" 
        )
    
    def get_algoritma_options():
        try:
            conn = connect_to_db()
            cursor = conn.cursor()
            cursor.execute("SELECT nama_algoritma FROM algoritma ORDER BY tanggal_dibuat DESC")
            results = cursor.fetchall()
            options = [row[0] for row in results]
            return options
        except mysql.connector.Error as e:
            st.error(f"Gagal mengambil data algoritma: {e}")
            return []
        finally:
            if 'cursor' in locals(): cursor.close()
            if 'conn' in locals(): conn.close()
    
    def simpan_riwayat_ke_db(data):
        db_connection = connect_to_db()
        cursor = db_connection.cursor()

        query = """
            INSERT INTO riwayat_prediksi (
                id_test, nim, nama, ipk, lama_studi, skill_kerjasama, beasiswa, hasil, tanggal, model_digunakan, id_pengguna
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            data["id_test"],
            data["nim"],
            data["nama"],
            data["ipk"],
            data["lama_studi"],
            data["skill_kerjasama"],
            data["beasiswa"],
            data["hasil"],
            data["tanggal"],
            data["model_digunakan"],
            data["id_pengguna"]
        )

        try:
            cursor.execute(query, values)
            db_connection.commit()
            cursor.close()
            db_connection.close()
            # st.success("Riwayat telah disimpan ke database.")
        except mysql.connector.Error as err:
            st.error(f"Gagal menyimpan data: {err}")
            cursor.close()
            db_connection.close()

    # menghapus riwayat by ID
    def hapus_riwayat_from_db(id_test):
        db_connection = connect_to_db()
        cursor = db_connection.cursor()

        query = "DELETE FROM riwayat_prediksi WHERE id_test = %s"
        try:
            cursor.execute(query, (id_test,))
            db_connection.commit()
            cursor.close()
            db_connection.close()
            st.success(f"Riwayat dengan ID {id_test} telah dihapus dari database.")
            return True
        except mysql.connector.Error as err:
            st.error(f"Gagal menghapus data: {err}")
            cursor.close()
            db_connection.close()
            return False

    # menampilkan detail hasil
    def tampilkan_detail_from_db(row):
        # Membuat label untuk expander menggunakan NIM, Nama, dan Hasil Prediksi
        expander_label = f"{row['nim']} - {row['nama']} - {row['hasil']}"

        # Menggunakan expander untuk menampilkan detail
        with st.expander(expander_label):
            st.write(f"**ID Test**: {row['id_test']}")
            st.write(f"**Nama Mahasiswa**: {row['nama']}")
            st.write(f"**NIM**: {row['nim']}")
            st.write(f"**Kategori IPK**: {row['ipk']}")
            st.write(f"**Kategori Lama Studi**: {row['lama_studi']}")
            st.write(f"**Skill Kerjasama**: {row['skill_kerjasama']}")
            st.write(f"**Beasiswa**: {row['beasiswa']}")
            st.write(f"**Hasil Prediksi**: {row['hasil']}")
            st.write(f"**Model Testing**: {row['model_digunakan']}")
            st.write(f"**Tanggal**: {row['tanggal']}")

    def tampilkan_riwayat_prediksi(sort_option="Terbaru", search_query=""):
        db_connection = connect_to_db()
        cursor = db_connection.cursor(dictionary=True)

        query = "SELECT * FROM riwayat_prediksi WHERE id_pengguna = %s"
        cursor.execute(query, (st.session_state["id_pengguna"],))
        df_history = pd.DataFrame(cursor.fetchall())

        # Jika tidak ada data riwayat
        if df_history.empty:
            st.info("😅 Upss... kamu belum pernah melakukan prediksi nih.")
        else:
            # Konversi kolom tanggal jika ada kolom bernama 'tanggal'
            if 'tanggal' in df_history.columns:
                df_history['tanggal'] = pd.to_datetime(df_history['tanggal'], errors='coerce', dayfirst=True)

            # Terapkan urutan sesuai pilihan dropdown
            if sort_option == "Abjad":
                df_history = df_history.sort_values(by="nama", ascending=True)
            elif sort_option == "NIM":
                df_history = df_history.sort_values(by="nim", ascending=True)
            elif sort_option == "Terbaru":
                df_history = df_history.sort_values(by="tanggal", ascending=False)
            elif sort_option == "Terlama":
                df_history = df_history.sort_values(by="tanggal", ascending=True)

            # Kolom pencarian & caption
            col10, col11 = st.columns([4, 2])
            with col10:
                search_query = st.text_input("Pencarian", placeholder="Cari NIM atau Nama Mahasiswa", label_visibility="hidden")
                try:
                    jumlah_riwayat = df_history.shape[0]
                    st.caption(f"Data Tersimpan: **{jumlah_riwayat}**")
                except:
                    st.caption("⚠️ Gagal membaca data.")

        st.markdown("<hr>", unsafe_allow_html=True)
        # Scrollable container dimulai
        st.markdown("""
            <style>
            .scroll-container {
                max-height: 500px;
                overflow-y: auto;
                padding-right: 10px;
            }
            </style>
            <div class="scroll-container">
        """, unsafe_allow_html=True)

        # Pencarian
        if search_query:
            filtered_df = df_history[df_history['nim'].astype(str).str.contains(search_query, case=False) |
                                    df_history['nama'].str.contains(search_query, case=False)]
            if filtered_df.empty:
                st.warning("Tidak ditemukan data yang dicari.")
            else:
                for index, row in filtered_df.iterrows():
                    col1, col2, col3 = st.columns([14, 2, 1])
                    with col1:
                        tampilkan_detail_from_db(row)
                    with col2:
                        if st.button(f"Hapus", key=row['id_test']):
                            if hapus_riwayat_from_db(row['id_test']):
                                st.rerun()
        else:
            for index, row in df_history.iterrows():
                col1, col2, col3 = st.columns([14, 2, 1])
                with col1:
                    tampilkan_detail_from_db(row)
                with col2:
                    if st.button(f"Hapus", key=row['id_test']):
                        if hapus_riwayat_from_db(row['id_test']):
                            st.rerun()

        # Scroll container selesai
        st.markdown("</div>", unsafe_allow_html=True)
        
        cursor.close()
        db_connection.close()

    def load_model(nama_algoritma, host="localhost", user="root", password="", database="data_mining_system"):
        try:
            # Koneksi database
            conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            cursor = conn.cursor()

            # Ambil model dari database berdasarkan nama_algoritma
            sql_model = """
                SELECT file_model FROM models 
                WHERE nama_algoritma = %s 
                ORDER BY tanggal_dibuat DESC 
                LIMIT 1
            """
            cursor.execute(sql_model, (nama_algoritma,))
            result_model = cursor.fetchone()

            if result_model is None:
                st.markdown("<div style='padding-top: 20rem;'>", unsafe_allow_html=True)
                st.markdown("""
                <div style='padding:10px; background-color:#fff3cd; border-left:6px solid #ffeeba; border-radius:4px'>
                    😅 <strong>Upss kamu belum pernah membuat model nih..</strong><br>
                    Buat modelmu dulu pada layanan <strong>Training Model</strong> 😊
                </div>
                """, unsafe_allow_html=True)
                return (None,) * 7

            # Load model dari binary
            model_binary = result_model[0]
            loaded_model = joblib.load(io.BytesIO(model_binary))

            # Fungsi bantu load encoder berdasarkan kode_encoder
            def load_encoder_from_db(kode_encoder):
                sql_enc = "SELECT file_encoder FROM encoder WHERE kode_encoder = %s LIMIT 1"
                cursor.execute(sql_enc, (kode_encoder,))
                result = cursor.fetchone()
                return joblib.load(io.BytesIO(result[0])) if result else None

            # Load semua encoder
            le_ipk = load_encoder_from_db('ENC001')
            le_studi = load_encoder_from_db('ENC002')
            le_beasiswa = load_encoder_from_db('ENC003')
            le_kerjatim = load_encoder_from_db('ENC004')
            le_ketereratan = load_encoder_from_db('ENC005')
            scaler = load_encoder_from_db('SCL001')  # opsional, jika tidak ada abaikan saja

            return loaded_model, le_ipk, le_studi, le_kerjatim, le_beasiswa, le_ketereratan, scaler

        except mysql.connector.Error as err:
            st.warning(f"❌ Kesalahan koneksi ke database: {err}")
            return (None,) * 7
        except Exception as e:
            st.warning(f"❌ Gagal memuat model/encoder: {e}")
            return (None,) * 7
        finally:
            cursor.close()
            conn.close()
# Fungsi untuk melakukan SMOTE pada data
    def apply_smote(X, y):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        return X_resampled, y_resampled
    
    def diagram_distribusi_data(original_labels, resampled_labels):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Ambil label unik untuk menentukan warna
        unique_labels = sorted(original_labels.unique())
        colors_before = ['steelblue', 'tomato']
        colors_after = ['steelblue', 'tomato']

        # Distribusi sebelum SMOTE
        counts_before = original_labels.value_counts()
        axes[0].bar(counts_before.index.astype(str), counts_before.values, color=colors_before)
        axes[0].set_title('Distribusi Kelas Sebelum SMOTE')
        axes[0].set_xlabel('Ketereratan', fontsize=14)
        axes[0].set_ylabel('Jumlah', fontsize=14)

        # Distribusi setelah SMOTE
        counts_after = resampled_labels.value_counts()
        axes[1].bar(counts_after.index.astype(str), counts_after.values, color=colors_after)
        axes[1].set_title('Distribusi Kelas Setelah SMOTE')
        axes[1].set_xlabel('Ketereratan', fontsize=14)
        axes[1].set_ylabel('Jumlah', fontsize=14)

        plt.tight_layout()
        st.pyplot(fig)  # Menampilkan grafik di Streamlit

    
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return encoded

    def get_model_options_from_db(
        host="localhost",
        user="root",
        password="",
        database="data_mining_system"
    ):
        try:
            conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            cursor = conn.cursor()
            query = "SELECT nama_algoritma FROM models WHERE id_pengguna = %s"
            cursor.execute(query, (st.session_state["id_pengguna"],))
            result = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return result

        except mysql.connector.Error as err:
            st.warning(f"❌ Gagal mengambil data model dari database: {err}")
            return []
    
    def get_user_data(user_id):
        db = connect_to_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM pengguna WHERE id_pengguna=%s", (user_id,))
        user = cursor.fetchone()
        cursor.close()
        db.close()
        return user

    CSS.cssAPP()
    col01, col02, col03, col04 =st.columns([10, 0.7, 0.7, 0.01])
    with col01:
        # Judul Aplikasi
        st.markdown(
        """
        <div style='padding-top: 0.5rem;'>
            <h1 style='color: #2657A3; margin-bottom: 0;'>CAREER RELEVANCE ANALYSIS</h1>
        </div>
        """,
        unsafe_allow_html=True
        )
    with col02 :
        user = st.session_state.user
        user_id = user["id_pengguna"]
        user_data = get_user_data(user_id)
        st.markdown("<div style='padding-top: 1rem;'>", unsafe_allow_html=True)
        if user_data["foto_profil"]:
            encoded_img = base64.b64encode(user_data["foto_profil"]).decode()
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <img src="data:image/jpeg;base64,{encoded_img}" 
                        style="width: 70px; height: 70px; border-radius: 50%; border: 2px solid #2657A3; margin-bottom: 1rem;">
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <img src="https://www.w3schools.com/howto/img_avatar.png" 
                        style="width: 70px; height: 70px; border-radius: 50%; border: 2px solid #2657A3;">
                </div>
                """, 
                unsafe_allow_html=True
            )
    with col03 :
        nama_lengkap = st.session_state.user['nama_lengkap']
        nama_terakhir = nama_lengkap.strip().split()[-1]
        if len(nama_terakhir) > 7:
            nama_tampil = nama_terakhir[:7] + "*"
        else:
            nama_tampil = nama_terakhir
        st.write(f"Hai, {nama_tampil}")
        if st.button("Logout"):
            login.logout()
            st.session_state.page = "login"
            st.rerun()
            
    st.markdown("<div style='padding-top: 1rem;'>", unsafe_allow_html=True)

    #cek hak akses
    hak_akses = st.session_state.user['hak_akses']
    # st.sidebar.header("Menu")

    with st.sidebar:
        logo_path = "logo/logo_sistem_prediksi_utama.png"
        encoded_logo = get_base64_image(logo_path)

        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded_logo}" width="230">
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<div style='padding-top: 1.8rem;'>", unsafe_allow_html=True)

    if hak_akses == 'admin':
        with st.sidebar:
            menu = option_menu(
                menu_title="Menu", 
                options=["Dashboard","Mining Data", "Riwayat Prediksi", "Kelola Pengguna","Kelola Algoritma","Kelola Model","Kelola Notifikasi","Import Model","Profil Saya"],
                icons=["house-fill","bar-chart", "clock-history", "person-lines-fill", "cpu", "diagram-3-fill","bell-fill", "cloud-arrow-down", "person"],
                default_index=0,
                menu_icon="list",
            )
    else:
        with st.sidebar:    
            menu = option_menu(
                menu_title="Menu",
                options=["Dashboard","Mining Data","Riwayat Prediksi","Import Model","Profil Saya"],
                icons=["house-fill","bar-chart", "clock-history", "cloud-arrow-down","person"],
                default_index=0,
                menu_icon="list"
            )

    if menu == "Dashboard":

        if 'edit_profil' in st.session_state:
            del st.session_state['edit_profil']

        if 'foto_profil_preview' in st.session_state:
            del st.session_state['foto_profil_preview']

        if 'nada_preview' in st.session_state:
            del st.session_state['nada_preview']
            
        if 'edit_notif' in st.session_state:
            del st.session_state['edit_notif']

        if 'hal_arsip_model' in st.session_state:
            del st.session_state["hal_arsip_model"]

        dashboard.show_dashboard()

    elif menu == "Riwayat Prediksi":

        if 'edit_profil' in st.session_state:
            del st.session_state['edit_profil']

        if 'foto_profil_preview' in st.session_state:
            del st.session_state['foto_profil_preview']
        
        if 'nada_preview' in st.session_state:
            del st.session_state['nada_preview']
            
        if 'edit_notif' in st.session_state:
            del st.session_state['edit_notif']

        if 'hal_arsip_model' in st.session_state:
            del st.session_state["hal_arsip_model"]

        col_header, col_sort, col_space = st.columns([7, 1,1.2])
        with col_header:
            st.subheader("Riwayat Prediksi")
            st.markdown("Berikut adalah daftar model yang tersimpan dalam sistem.")
        with col_sort:
            st.caption("Sort by:")
            sort_option = st.selectbox(
                "Sort by",
                options=["Abjad", "NIM", "Terbaru", "Terlama"],
                index=2,
                label_visibility="collapsed"
            )
        tampilkan_riwayat_prediksi(sort_option, search_query="")

    elif menu == "Kelola Pengguna":

        if 'edit_profil' in st.session_state:
            del st.session_state['edit_profil']

        if 'foto_profil_preview' in st.session_state:
            del st.session_state['foto_profil_preview']

        if 'nada_preview' in st.session_state:
            del st.session_state['nada_preview']
            
        if 'edit_notif' in st.session_state:
            del st.session_state['edit_notif']

        if "halaman" not in st.session_state:
            st.session_state.halaman = "kelola_pengguna"

        if 'hal_arsip_model' in st.session_state:
            del st.session_state["hal_arsip_model"]

        # Navigasi berdasarkan state halaman
        if st.session_state.halaman == "kelola_pengguna":
            kelola_pengguna.main()

        elif st.session_state.halaman == "form_tambah_pengguna":
            form_tambah_pengguna.form_tambah_pengguna()

    elif menu == "Kelola Algoritma":
        if 'edit_profil' in st.session_state:
            del st.session_state['edit_profil']

        if 'foto_profil_preview' in st.session_state:
            del st.session_state['foto_profil_preview']

        if 'nada_preview' in st.session_state:
            del st.session_state['nada_preview']

        if 'edit_notif' in st.session_state:
            del st.session_state['edit_notif']

        if "hal_kelola_algoritma" not in st.session_state:
            st.session_state.hal_kelola_algoritma = "kelola_algoritma"

        if st.session_state.hal_kelola_algoritma == "kelola_algoritma":
            kelola_algoritma.main()

        elif st.session_state.hal_kelola_algoritma == "tambah_algoritma":
            tambah_algoritma.main()

        if 'hal_arsip_model' in st.session_state:
            del st.session_state["hal_arsip_model"]

    elif menu == "Kelola Model":

        if 'edit_profil' in st.session_state:
            del st.session_state['edit_profil']

        if 'foto_profil_preview' in st.session_state:
            del st.session_state['foto_profil_preview']

        if 'nada_preview' in st.session_state:
            del st.session_state['nada_preview']
            
        if 'edit_notif' in st.session_state:
            del st.session_state['edit_notif']

        if "hal_kelola_model" not in st.session_state:
            st.session_state.hal_kelola_model = "kelola_model"

        if st.session_state.hal_kelola_model == "kelola_model":
            kelola_model.main()

        elif st.session_state.hal_kelola_model == "detail_model":
            detail_model.main()

        if 'hal_arsip_model' in st.session_state:
            del st.session_state["hal_arsip_model"]

    elif menu == "Import Model":

        if 'edit_profil' in st.session_state:
            del st.session_state['edit_profil']

        if 'foto_profil_preview' in st.session_state:
            del st.session_state['foto_profil_preview']

        if 'nada_preview' in st.session_state:
            del st.session_state['nada_preview']
            
        if 'edit_notif' in st.session_state:
            del st.session_state['edit_notif']

        if "import_model" not in st.session_state:
            st.session_state.import_model = "show"

        if st.session_state.import_model == "show":
            import_model.main()

        if 'hal_arsip_model' in st.session_state:
            del st.session_state["hal_arsip_model"]
    
    elif menu == "Kelola Notifikasi":
        
        if 'edit_profil' in st.session_state:
            del st.session_state['edit_profil']

        if 'foto_profil_preview' in st.session_state:
            del st.session_state['foto_profil_preview']

        if 'nada_preview' in st.session_state:
            del st.session_state['nada_preview']
            
        if 'edit_notif' in st.session_state:
            del st.session_state['edit_notif']
        
        if "hal_kelola_notif" not in st.session_state:
            st.session_state.hal_kelola_notif = "kelola_notif"

        if st.session_state.hal_kelola_notif == "kelola_notif":
            kelola_notif.main()

        if 'hal_arsip_model' in st.session_state:
            del st.session_state["hal_arsip_model"]

    elif menu == "Profil Saya":
        
        if "hal_profil" not in st.session_state:
            st.session_state.hal_profil = "kelola_profil"

        if st.session_state.hal_profil == "kelola_profil":
            profil.main()

        if 'hal_arsip_model' in st.session_state:
            del st.session_state["hal_arsip_model"]
        
    elif menu == "Mining Data":
        if 'edit_profil' in st.session_state:
            del st.session_state['edit_profil']

        if 'foto_profil_preview' in st.session_state:
            del st.session_state['foto_profil_preview']

        if 'nada_preview' in st.session_state:
            del st.session_state['nada_preview']
            
        if 'edit_notif' in st.session_state:
            del st.session_state['edit_notif']
            
        fitur = st.sidebar.selectbox("Pilih Layanan", ["Prediksi", "Training Model","Sampling Data"], index=1)

        # --- Ambil ID Unik ---
        def generate_unique_id():
            # Membuat ID unik menggunakan UUID
            unique_id = f"PRD{str(uuid.uuid4().int)[:8]}"  # Ambil 8 karakter pertama dari UUID
            return unique_id

        if fitur == "Prediksi" :

            if "reset_form" not in st.session_state:
                st.session_state.reset_form = False

            if "input_nim" not in st.session_state:
                st.session_state.input_nim = ""

            if "input_nama" not in st.session_state:
                st.session_state.input_nama = ""

            # Reset nilai setelah rerun
            if st.session_state.reset_form:
                st.session_state.input_nim = ""
                st.session_state.input_nama = ""
                st.session_state.reset_form = False

            st.markdown("""
            <div style="
                background-color: #2657A3;
                padding: 1px 0;
                width: 100%;
                overflow: hidden;
                position: relative;
            ">
                <marquee behavior="scroll" direction="left" scrollamount="5" style="
                    color: white;
                    font-weight: bold;
                    font-size: 20px;
                ">
                    Layanan Prediksi Ketereratan Karier
                </marquee>
            </div>
        """, unsafe_allow_html=True)
            
            #pilih model yang mau digunakan
            model_options = get_model_options_from_db()
            if model_options:
                pilihmodel = st.sidebar.selectbox("Pilih Model Prediksi", model_options, index=0)
            else:
                st.sidebar.warning("⚠️ Tidak ada model yang tersedia di database.")
                pilihmodel = None
            
            with st.sidebar.expander("📄 Riwayat Prediksi"):
                show_history = st.checkbox("Tampilkan Riwayat Prediksi")
            # Load model dan objek sesuai pilihan model
            model, le_ipk, le_studi, le_kerjatim, le_beasiswa, le_ketereratan, scaler = load_model(pilihmodel)

            if model is None or le_ipk is None or scaler is None:
                st.stop()

            # 🔹 Form inputan pengguna (REVISI SESUAI PERMINTAAN)
            # st.subheader("Masukkan Data untuk Prediksi Ketereratan")

            nim = st.text_input("NIM Mahasiswa", key="input_nim", placeholder="Masukkan NIM (9 digit)")
            if nim and (not nim.isdigit() or len(nim) != 9):
                st.warning("NIM harus terdiri dari 9 digit angka.")

            nama_mahasiswa = st.text_input(
                "Nama Mahasiswa",
                key="input_nama",
                placeholder="Masukkan nama lengkap mahasiswa"
            ).upper()

            #----------------INPUT IPK & LAMA STUDI--------------------------

            col1, mid, col2, col21, col22, col23 = st.columns([0.3, 0.01, 0.3,0.1,0.3, 0.3])
            with col1:
                pembilang = st.number_input("IPK (Skala 4)", min_value=1, max_value=4, step=1, format="%d")
            with mid:
                st.write("")
                st.write("")
                st.markdown("<h4 style='text-align: left;'>,</h4>", unsafe_allow_html=True)
            with col2:
                if pembilang == 4:
                    penyebut = st.number_input("penyebut", min_value=0, max_value=0, step=0, format="%02d", disabled=True, label_visibility="hidden")
                else:
                    penyebut = st.number_input("penyebut", min_value=0, max_value=99, step=1, format="%02d", label_visibility="hidden")
            with col22:
                tahun = st.number_input("Lama Studi", min_value=1, max_value=7, step=1, format="%d")
                st.caption("Tahun")
            with col23:
                if tahun == 7:
                    bulan = st.number_input("bulan", min_value=0, max_value=0, step=0, format="%d", disabled=True, label_visibility="hidden")
                    st.caption("Bulan")
                else:
                    bulan = st.number_input("bulan", min_value=0, max_value=11, step=1, format="%d", label_visibility="hidden")
                    st.caption("Bulan")

            #----------------INPUT KERJA TIM--------------------------

            col24, col25, col26= st.columns([0.53, 0.08, 0.51])
            with col24:
                kerja_tim_input = st.selectbox(
                    "Kemampuan Kerja Tim",
                    options=["-- Pilih --","SANGAT MAMPU", "MAMPU", "CUKUP", "KURANG", "SANGAT KURANG"]
                )
            with col26:
                st.write("")
                beasiswa_status = st.radio(
                    "Pernah Mendapatkan Beasiswa?",
                    options=["Pernah", "Belum Pernah"],
                    index=1
                )

            # 🔹 Tombol untuk jalankan prediksi
            if st.button("Jalankan Prediksi"):
                if nim == "":
                    warning_nim = st.empty()
                    warning_nim.warning("NIM mahasiswa wajib diisi.")
                    time.sleep(5)
                    warning_nim.empty()
                elif nama_mahasiswa == "":
                    warning_nama = st.empty()
                    warning_nama.warning("Nama mahasiswa wajib diisi.")
                    time.sleep(5)
                    warning_nama.empty()
                elif kerja_tim_input == "-- Pilih --":
                    warning_kerjatim = st.empty()
                    warning_kerjatim.warning("Silakan pilih kemampuan kerja tim terlebih dahulu.")
                    time.sleep(5)
                    warning_kerjatim.empty()
                # ----------------------------
                # 🔥 Proses konversi ke kategori
                # ----------------------------
                else:
                #IPK asli
                    ipk_asli = float(f"{pembilang}.{penyebut:02d}") 
                    # Kategori IPK
                    ipk_combined = int(f"{pembilang}{penyebut:02d}")
                    if 0 <= ipk_combined <= 300:
                        kategori_ipk = "RENDAH"
                    elif 301 <= ipk_combined <= 350:
                        kategori_ipk = "MEDIAN"
                    else:
                        kategori_ipk = "TINGGI"
                    
                    #lamastudi
                    lamastudi_asli = f"{tahun} Tahun {bulan} Bulan"
                    # Kategori Lama Studi
                    lama_studi_bulan = (tahun * 12) + bulan
                    if 0 <= lama_studi_bulan <= 48:
                        kategori_lama_studi = "TEPAT"
                    elif 49 <= lama_studi_bulan <= 54:
                        kategori_lama_studi = "CUKUP"
                    else:
                        kategori_lama_studi = "LAMA"

                    # # Kerja Tim
                    # kerja_tim = "SANGAT KURANG" if kerja_tim_input == 1 else \
                    # "KURANG" if kerja_tim_input == 2 else \
                    # "CUKUP" if kerja_tim_input == 3 else \
                    # "MAMPU" if kerja_tim_input == 4 else \
                    # "SANGAT MAMPU" if kerja_tim_input == 5 else \
                    # "TIDAK DIKETAHUI"

                    # Pernah Kerja
                    beasiswa = "YA" if beasiswa_status == "Pernah" else "TIDAK"

                    # ----------------------------
                    # 🔥 Proses input ke dataframe
                    # ----------------------------
                    input_data = {
                        'kategori_ipk': [kategori_ipk],
                        'kategori_lama_studi': [kategori_lama_studi],
                        'kerja tim': [kerja_tim_input],
                        'beasiswa': [beasiswa]
                    }
                    input_df = pd.DataFrame(input_data)

                    # Label encoding (gunakan encoder yang sesuai)
                    input_df['kategori_ipk'] = le_ipk.transform(input_df['kategori_ipk'])
                    input_df['kategori_lama_studi'] = le_studi.transform(input_df['kategori_lama_studi'])
                    input_df['kerja tim'] = le_kerjatim.transform(input_df['kerja tim'])
                    input_df['beasiswa'] = le_beasiswa.transform(input_df['beasiswa'])

                    if pilihmodel == "Naive Bayes":
                        prediction = model.predict(input_df)
                    else:
                        # Normalisasi
                        input_scaled = scaler.transform(input_df)
                        # Prediksi
                        prediction = model.predict(input_scaled)

                    predicted_label = le_ketereratan.inverse_transform(prediction)

                    # Pastikan NIM disimpan sebagai string
                    nim = str(nim)

                    # Format hasil prediksi agar tidak tersimpan sebagai array
                    hasil_prediksi = predicted_label[0]

                    with st.spinner('Harap tunggu, memproses prediksi...'):
                        time.sleep(4)  

                    # Tampilkan hasil
                    if predicted_label[0].upper() == "ERAT":
                        st.success(f"Prediksi karier : **{predicted_label[0]}**")
                    else:
                        st.error(f"Prediksi karier : **{predicted_label[0]}**")

                    # Simpan riwayat
                    new_row= {
                        "id_test": generate_unique_id(),
                        "nim": nim,
                        "nama": nama_mahasiswa,
                        "ipk": f"{ipk_asli} / {kategori_ipk}",
                        "lama_studi": f"{lamastudi_asli} / {kategori_lama_studi}",
                        "skill_kerjasama": kerja_tim_input,
                        "beasiswa": beasiswa,
                        "hasil": hasil_prediksi,
                        "model_digunakan": pilihmodel,  # ⬅️ Tambahkan ini
                        "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "id_pengguna": st.session_state.get("id_pengguna")
                    }

                    simpan_riwayat_ke_db(new_row)

                    time.sleep(5)
                    st.session_state.reset_form = True
                    st.rerun()
                
                # Tampilkan garis pembatas
                st.markdown("<hr>", unsafe_allow_html=True)
            
            # Tampilkan riwayat
        if fitur == "Prediksi" and show_history:
            st.markdown("<hr>", unsafe_allow_html=True)
            col_header, col_sort, col_space = st.columns([7, 1,1.2])
            with col_header:
                st.subheader("Riwayat Prediksi")
                st.markdown("Berikut adalah daftar model yang tersimpan dalam sistem.")
            with col_sort:
                st.caption("Sort by:")
                sort_option = st.selectbox(
                    "Sort by",
                    options=["Abjad", "NIM", "Terbaru", "Terlama"],
                    index=2,
                    label_visibility="collapsed"
                )
            tampilkan_riwayat_prediksi(sort_option, search_query="")



        #------------------------------------------------------------------------------------------------------
        elif fitur == "Training Model":

            st.markdown("""
            <div style="
                background-color: #2657A3;
                padding: 1px 0;
                width: 100%;
                overflow: hidden;
                position: relative;
            ">
                <marquee behavior="scroll" direction="left" scrollamount="5" style="
                    color: white;
                    font-weight: bold;
                    font-size: 20px;
                ">
                    Layanan Training Data Lulusan
                </marquee>
            </div>
        """, unsafe_allow_html=True)

            # Menu untuk lihat model yang tersimpan
            with st.sidebar.expander("Lihat Model Tersimpan (Database)"):
                colD,colE,colF = st.columns([1,2,8])
                with colD:
                # Tombol refresh
                    if st.button("↻"):
                        if 'hal_arsip_model' in st.session_state:
                            del st.session_state["hal_arsip_model"]
                        if 'info_mining' in st.session_state:
                            del st.session_state["info_mining"]
                        st.rerun() 
                with colF:
                    if 'hal_arsip_model' not in st.session_state:
                        if st.button("Model Terhapus"):
                            st.session_state.hal_arsip_model = "show"
                            if 'info_mining' not in st.session_state:
                                st.session_state.info_mining = "hide"
                            st.rerun()
                    else: 
                        st.button("Model Terhapus", key='btn_second')
                        
                try:
                    # Koneksi ke database
                    conn = mysql.connector.connect(
                        host="localhost",
                        user="root",
                        password="",
                        database="data_mining_system"
                    )
                    cursor = conn.cursor(dictionary=True)  # agar hasilnya dalam bentuk dict

                    # Ambil semua model yang tersimpan
                    query = """
                        SELECT kode_model, kode_algoritma, nama_algoritma, jumlah_fold, akurasi, tanggal_dibuat, id_pengguna
                        FROM models
                        WHERE id_pengguna = %s
                        ORDER BY tanggal_dibuat DESC
                    """
                    cursor.execute(query, (st.session_state["id_pengguna"],))
                    hasil = cursor.fetchall()

                    if hasil:
                        st.success(f"Ada {len(hasil)} model yang tersimpan:")
                        for model in hasil:
                            st.write(f"📂 **{model['nama_algoritma']}** (`{model['kode_model']}`) (`{model['akurasi'] * 100:.2f}%`)")

                            st.caption(
                                f"Dibuat: {model['tanggal_dibuat'].strftime('%d-%m-%Y %H:%M:%S')}"
                            )
                    else:
                        st.info("Belum ada model yang tersimpan di database.")

                except mysql.connector.Error as err:
                    st.error(f"❌ Gagal mengakses database: {err}")

                finally:
                    if 'cursor' in locals():
                        cursor.close()
                    if 'conn' in locals() and conn.is_connected():
                        conn.close() 

            if 'hal_arsip_model' in st.session_state:
                if st.session_state.hal_arsip_model == "show":
                    arsip_model.main()
                
        #------------------------------------------------------------------------------------------------------

            # Fungsi untuk memproses dataset PKPA
            def process_pkpa(file):
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                    if all(col in df.columns for col in ["Kode Progdi", "nim", "nama", "pekerjaan", "beasiswa","ketereratan", "kerja tim"]):
                        df = df[["Kode Progdi", "nim", "nama", "pekerjaan","beasiswa", "ketereratan", "kerja tim"]]
                    else:
                        st.error("ada kolom yang kurang")
                        return pd.DataFrame()
                else:
                    excel_file = pd.ExcelFile(file)
                    if "Rekap" in excel_file.sheet_names:
                        df = pd.read_excel(file, sheet_name="Rekap", usecols="C,D,E,T,AA,AC,AJ", skiprows=1)
                        df.columns = ["Kode Progdi", "nim", "nama", "pekerjaan", "beasiswa","ketereratan", "kerja tim"]                     # ganti nama kolom broo
                    else:
                        st.error("File Excel harus memiliki sheet 'Rekap'")
                        return pd.DataFrame()
                df = df.dropna()
                
                df['ketereratan'] = df['ketereratan'].astype(str).str.strip()
                df = df[df['ketereratan'].isin(['1', '2', '3', '4', '5'])]  # jaga-jaga, pastikan hanya yang valid
                df['ketereratan'] = df['ketereratan'].astype(int)

                df = df[~df['beasiswa'].astype(str).str.strip().isin(["0"])]  # Filter nilai "0"
                df['beasiswa'] = df['beasiswa'].astype(str).str.strip()  # Menghilangkan spasi tambahan
                df = df[df['beasiswa'].isin(['1', '2', '3', '4', '5', '6', '7'])]
                df['beasiswa'] = df['beasiswa'].apply(lambda x: 'YA' if x in ['2', '3', '4', '5', '6', '7'] else 'TIDAK')

                df = df[~df['kerja tim'].astype(str).str.strip().isin(["0"])]
                df['kerja tim'] = df['kerja tim'].astype(str).str.strip()
                df = df[df['kerja tim'].isin(['1', '2', '3', '4', '5'])]
                df['kerja tim'] = df['kerja tim'].astype(int)
                
                # df['ormawa'] = df['ormawa'].astype(str).str.strip()
                # df = df[df['ormawa'].isin(['0', '1', '2', '3', '4'])]
                # df['ormawa'] = df['ormawa'].astype(int)
                
                df['ketereratan'] = df['ketereratan'].apply(lambda x: 'ERAT' if x in [1, 2, 3] else 'TIDAK ERAT')
                df['kerja tim'] = df['kerja tim'].apply(lambda x: 
                    'SANGAT KURANG' if x == 1 else
                    'KURANG' if x == 2 else
                    'CUKUP' if x == 3 else 
                    'MAMPU' if x == 4 else 
                    'SANGAT MAMPU' if x == 5 else
                    'TIDAK DIKETAHUI' 
                )

                # df['ormawa'] = df['ormawa'].apply(lambda x: 
                #     'TIDAK MENGIKUTI' if x == 0 else
                #     '1' if x == 1 else
                #     '2' if x == 2 else 
                #     '3' if x == 3 else 
                #     'LEBIH DARI 3' if x == 4 else
                #     'TIDAK DIKETAHUI' 
                # )

                df = df[~df['Kode Progdi'].isin(['01', '02', '03'])]                                                           #filter example without
                df['nim'] = df['nim'].astype(str).str.strip()
                df = df[df['nim'].str.isdigit()]
                df = df[df['nim'].str.len() == 9]


                return df[["nim","beasiswa","ketereratan", "kerja tim"]]
            
            # Fungsi untuk memproses dataset BAAK
            def process_baak(file):
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    excel_file = pd.ExcelFile(file)
                    processed_sheets = []
                    for sheet in excel_file.sheet_names:
                        df = pd.read_excel(file, sheet_name=sheet, usecols="B,C,D,E", skiprows=1)           #select attribute
                        df.columns = ["nim", "nama", "lama studi", "ipk"]                                  #buat kolom tabel
                        
                        df = df.dropna()                                
                        df = df[~df['nim'].apply(lambda x: str(x)[4:6] in ['01', '02', '03'])]              # filter example
                        
                        # Konversi nilai di kolom "lama studi" dari format tahun ke format bulan
                        def convert_lama_studi(value):
                            value = str(value)                                                          #memisah karakter per spasi
                            if '.' in value:
                                bagian_depan, bagian_belakang = value.split('.')
                                bulan = int(bagian_depan) * 12 + int(bagian_belakang)
                            else:
                                bulan = int(value) * 12
                            return bulan
                        
                        df["lama studi"] = df["lama studi"].apply(convert_lama_studi)               #isi kolom lama studi dengan nilai hasil konversi ke bulan
                        
                        # Mengalikan nilai di kolom "ipk" dengan 100
                        df["ipk"] = df["ipk"] * 100                                                 #isi kolom ipk dengan nilai ipk konversi x 100
                        
                        # Filter hanya data yang memiliki lama studi antara 40 bulan dan 84 bulan serta ipk antara 250 dan 400
                        df = df[(df["lama studi"] >= 42) & (df["lama studi"] <= 60)]  # Filter lama studi
                        df = df[(df["ipk"] >= 250) & (df["ipk"] <= 400)]  # Filter ipk

                    #---------------------------------------------------
                        # Kategorisasi untuk kolom "ipk"
                        def categorize_ipk(ipk_value):
                            if 250 <= ipk_value <= 300:
                                return 'RENDAH'
                            elif 301 <= ipk_value <= 350:
                                return 'MEDIAN'
                            elif 351 <= ipk_value <= 400:
                                return 'TINGGI'
                            else:
                                return None

                        df['kategori_ipk'] = df['ipk'].apply(categorize_ipk)  # Menambahkan kolom kategori_ipk
                        
                        # Kategorisasi untuk kolom "lama studi"
                        def categorize_lama_studi(lama_studi_value):
                            if 42 <= lama_studi_value <= 48:
                                return 'TEPAT'
                            elif 49 <= lama_studi_value <= 54:
                                return 'CUKUP'
                            elif 55 <= lama_studi_value <= 60:
                                return 'LAMA'
                            else:
                                return None

                        df['kategori_lama_studi'] = df['lama studi'].apply(categorize_lama_studi)  # Menambahkan kolom kategori_lama_studi
                    #---------------------------------------------------
                        
                        processed_sheets.append(df)
                    df = pd.concat(processed_sheets, ignore_index=True) if processed_sheets else pd.DataFrame()

                    # Normalisasi NIM
                    df['nim'] = df['nim'].astype(str).str.strip()
                    df = df[df['nim'].str.isdigit()]
                    df = df[df['nim'].str.len() == 9]
                return df

            st.sidebar.header("Upload Dataset")
            uploaded_pkpa = st.sidebar.file_uploader("Upload dataset PKPA (Excel atau CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
            uploaded_baak = st.sidebar.file_uploader("Upload dataset BAAK (Excel atau CSV)", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
            jmlFold = st.sidebar.number_input(
                "Masukkan jumlah fold Cross Validation:",
                min_value=2, max_value=25, value=12, step=1
            )
            
            algoritma_options = get_algoritma_options()
            algoritma_select = st.sidebar.selectbox(
                "Pilih Algoritma",
                options=["--- Pilih Algoritma ---"] + algoritma_options,
                index=0,
                format_func=lambda x: "⬇ Pilih Algoritma" if x == "--- Pilih Algoritma ---" else x
            )

            if not (uploaded_pkpa and uploaded_baak and jmlFold and algoritma_select != "--- Pilih Algoritma ---"):
                st.markdown("<div style='padding-top: 20rem;'>", unsafe_allow_html=True)
                if 'info_mining' not in st.session_state:
                    st.info("Gunakan sidebar untuk mulai mengunggah data dan menjalankan model prediksi.")
            if uploaded_pkpa and uploaded_baak and jmlFold and algoritma_select != "--- Pilih Algoritma ---":
                st.markdown("<div style='padding-top: 1.9rem;'>", unsafe_allow_html=True)
                with st.spinner('Data sedang di konversi..'):
                    time.sleep(1)
                    df_pkpa_list = [process_pkpa(file) for file in uploaded_pkpa]
                    df_pkpa = pd.concat(df_pkpa_list, ignore_index=True)
                    st.write("Dataset PKPA yang diupload:")
                    st.write(df_pkpa)

                    df_baak_list = [process_baak(file) for file in uploaded_baak]
                    df_baak = pd.concat(df_baak_list, ignore_index=True)
                    st.write("Dataset BAAK yang diupload:")
                    st.write(df_baak)
                
                with st.spinner('Menggabungkan dataset...'):
                    time.sleep(1)
                    df_merged = df_pkpa.merge(df_baak, on="nim", how="left")
                    dataset = df_merged[["kategori_ipk", "kategori_lama_studi","beasiswa", "kerja tim", "ketereratan"]]
                    
                    dataset_filtered = pd.concat([
                        dataset[(dataset["ketereratan"].notna()) & dataset["kategori_ipk"].notna() & dataset["kategori_lama_studi"].notna() & 
                                (dataset["kategori_lama_studi"]) & (dataset["kategori_lama_studi"]) & dataset["kerja tim"].notna() & dataset["beasiswa"].notna()]
                    ], ignore_index=True)

                with st.spinner('SMOTE & Resampling dataset...'):

                    X = dataset_filtered[['kategori_ipk','kategori_lama_studi','beasiswa', 'kerja tim']]  # Fitur yang digunakan
                    y = dataset_filtered['ketereratan']  # Target yang digunakan

                    st.markdown(
                        "<h3 style='text-align: center;'>Distribusi Kelas</h3>",
                        unsafe_allow_html=True
                    )
                    col30, col40, col50 = st.columns([0.15,0.7, 0.5])
                    with col40:
                        st.write("Sebelum SMOTE:")
                        st.write(y.value_counts())

                    # Konversi YA/TIDAK menjadi 1/0
                    X['beasiswa'] = X['beasiswa'].replace({'YA': 1, 'TIDAK': 0})
                    X['kerja tim'] = X['kerja tim'].replace({
                        'SANGAT KURANG': 0,
                        'KURANG': 1,
                        'CUKUP': 2,
                        'MAMPU': 3,
                        'SANGAT MAMPU': 4
                    })

                    # X['ormawa'] = X['ormawa'].replace({
                    #     'TIDAK MENGIKUTI': 0,
                    #     '1': 1,
                    #     '2': 2,
                    #     '3': 3,
                    #     'LEBIH DARI 3': 4
                    # })
                    
                    X['kategori_ipk'] = X['kategori_ipk'].replace({'RENDAH': 0, 'MEDIAN': 1, 'TINGGI': 2})
                    X['kategori_lama_studi'] = X['kategori_lama_studi'].replace({'TEPAT': 0, 'CUKUP': 1, 'LAMA': 2})
                    y = y.replace({'ERAT': 1, 'TIDAK ERAT': 0})

                    # Terapkan SMOTE untuk Resampling
                    X_resampled, y_resampled = apply_smote(X, y)

                    # ⬅️ Simpan data sebelum mapping ulang
                    df_bahankorelasi = pd.DataFrame(X_resampled.copy())
                    df_bahankorelasi['ketereratan'] = y_resampled

                    with col50 :
                        st.write("Setelah SMOTE:")
                        st.write(pd.Series(y_resampled).value_counts())
                    diagram_distribusi_data(y, y_resampled)
                    

                    beasiswa_map = {1: 'YA', 0: 'TIDAK'}
                    kerja_tim_map = {0: 'SANGAT KURANG', 1: 'KURANG', 2: 'CUKUP', 3:'MAMPU', 4:'SANGAT MAMPU'}
                    # ormawa_map = {0: 'TIDAK MENGIKUTI', 1: '1', 2: '2', 3:'3', 4:'LEBIH DARI 3'}
                    ipk_map = {0: 'RENDAH', 1: 'MEDIAN', 2: 'TINGGI'}
                    lama_studi_map = {0: 'TEPAT', 1: 'CUKUP', 2: 'LAMA'}
                    ketereratan_map = {0: 'ERAT', 1: 'TIDAK ERAT'}

                    # Apply reverse mapping ke X_resampled
                    X_resampled['beasiswa'] = X_resampled['beasiswa'].map(beasiswa_map)
                    X_resampled['kerja tim'] = X_resampled['kerja tim'].map(kerja_tim_map)
                    # X_resampled['ormawa'] = X_resampled['ormawa'].map(ormawa_map)
                    X_resampled['kategori_ipk'] = X_resampled['kategori_ipk'].map(ipk_map)
                    X_resampled['kategori_lama_studi'] = X_resampled['kategori_lama_studi'].map(lama_studi_map)

                    # Apply reverse mapping ke y_resampled (pastikan y_resampled adalah Series, bukan array)
                    y_resampled_labels = pd.Series(y_resampled).map(ketereratan_map)

                    df_final = X_resampled.copy()
                    df_final['ketereratan'] = y_resampled_labels

                    st.write("Data hasil SMOTE:")
                    st.write(df_final)

                with st.expander("🔍 Korelasi Variabel Data Asli"):
                    try:
                        st.write("Data:")
                        st.dataframe(df_bahankorelasi)  # ⬅️ Cek dulu hasil merge

                        df_raw_corr = df_bahankorelasi[["beasiswa", "kerja tim", "kategori_lama_studi", "kategori_ipk", "ketereratan"]]

                        if df_raw_corr.empty:
                            st.warning("Data kosong setelah diproses. Cek apakah data masih numerik dan ada nilai.")
                        else:
                            # Korelasi Pearson
                            corr_raw = df_raw_corr.corr(method='pearson')

                            # Heatmap visual: tampilkan diagonal dan segitiga bawah saja
                            mask = np.triu(np.ones_like(corr_raw, dtype=bool), k=1)
                            plt.figure(figsize=(8, 6))
                            sns.heatmap(corr_raw,
                                        mask=mask,
                                        annot=True,
                                        cmap='coolwarm',
                                        vmin=-1, vmax=1,
                                        square=True,
                                        linewidths=0.5,
                                        fmt=".2f")
                            plt.xticks(rotation=45, ha='right', style='italic')
                            plt.yticks(rotation=0, style='italic')
                            plt.title("Korelasi Variabel (Pearson)")
                            st.pyplot(plt)

                            # Ambil korelasi X terhadap Y (ketereratan)
                            correlation_with_target = corr_raw['ketereratan'].drop('ketereratan')

                            correlation_result = []
                            for feature, corr_value in correlation_with_target.items():
                                arah = 'positif' if corr_value > 0 else 'negatif'
                                correlation_result.append({
                                    'Fitur': feature,
                                    'Korelasi': round(corr_value, 3),
                                    'Arah Korelasi': arah
                                })
                            correlation_df = pd.DataFrame(correlation_result).sort_values(by='Korelasi', key=abs, ascending=False)

                            st.write("📊 **Korelasi Pearson terhadap ketereratan**")
                            st.dataframe(correlation_df)

                    except Exception as e:
                        st.warning(f"❗ Gagal menghitung korelasi data asli: {e}")


                st.write("Data Siap Digunakan!")
                
                if algoritma_select == "Decision Tree":
                    
                    if st.button("Jalankan Decision Tree"):

                        # Catat waktu mulai
                        start_time = time.time()
                        results = run_decision_tree(df_final, n_splits=jmlFold)
                        # Catat waktu selesai
                        end_time = time.time()
                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Decision Tree</h3>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"""
                                <div style="background-color:#d1e7dd;padding:20px;border-radius:10px;">
                                    <h4 style="color:#0f5132;">🎯 Akurasi Model</h4>
                                    <h2 style="margin:0;color:#0f5132;">{results['accuracy'] * 100:.2f}%</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with col2:
                            st.markdown(
                                f"""
                                <div style="background-color:#cff4fc;padding:20px;border-radius:10px;">
                                    <h4 style="color:#055160;">⏱️ Waktu Komputasi</h4>
                                    <h2 style="margin:0;color:#055160;">{elapsed_time:.2f} detik</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        col33,col34=st.columns([0.5,0.5])
                        with col33:
                            st.write("**Confusion Matrix:**")
                            conf_matrix = results['confusion_matrix']
                            labels = ['TIDAK ERAT', 'ERAT']  # Ubah sesuai label datamu

                            fig, ax = plt.subplots()
                            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=labels, yticklabels=labels, cbar=True, ax=ax)
                            ax.set_xlabel('Predicted label')
                            ax.set_ylabel('True label')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                        with col34:
                            st.write("**Laporan Klasifikasi:**")
                            st.dataframe(results['classification_report'])
                            st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        col35, col36 = st.columns([0.5,0.5])
                        with col35:
                            st.write("**Precision:**", results['precision'])
                            st.write("**Recall:**", results['recall'])
                            st.write("**F1-Score:**", results['f1_score'])
                            st.write("**Specificity:**", results['specificity'])
                            st.write("**NPV:**", results['npv'])
                            st.write("***Mean Accuracy:***", results['average_accuracy'])
                        with col36:
                            st.write("**Cohen's Kappa:**", results['kappa'])
                            st.write("**ROC AUC Score:**", results["roc_auc"])
                            # st.write("**Fold Terbaik:**", results['fold_terbaik'])
                            st.write("***Mean Error:***", results['average_error'])
                            # st.write("**Data Training:**", results['jumlah_data_training'])
                            # st.write("**Data Testing:**", results['jumlah_data_testing'])
                        
                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        col371, col381,col382 = st.columns([0.5, 0.5, 0.5])
                        with col371:
                            st.image(results['max_depth_plot'], caption="Akurasi vs Max Depth")
                        with col381:
                            st.image(results['min_samples_split_plot'], caption="Akurasi vs Min Split")
                        with col382:
                            st.image(results['min_samples_leaf_plot'], caption="Akurasi vs Min Leaf")
                        st.write("**Visualisasi Decision Tree:**")
                        st.image(results['tree_image'], caption="Decision Tree", use_container_width=True)
                        st.write("**Feature Importance:**")
                        st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)

                elif algoritma_select == "Random Forest":

                    if st.button("Jalankan Random Forest"):
                            # Catat waktu mulai
                            start_time = time.time()
                            results = run_random_forest(df_final, n_splits=jmlFold)
                            # Catat waktu selesai
                            end_time = time.time()
                            # Hitung durasi komputasi
                            elapsed_time = end_time - start_time

                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown(
                                "<h3 style='text-align: center;'>Hasil Evaluasi Random Forest</h3>",
                                unsafe_allow_html=True
                            )
                            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown(
                                    f"""
                                    <div style="background-color:#d1e7dd;padding:20px;border-radius:10px;">
                                        <h4 style="color:#0f5132;">🎯 Akurasi Model</h4>
                                        <h2 style="margin:0;color:#0f5132;">{results['accuracy'] * 100:.2f}%</h2>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            with col2:
                                st.markdown(
                                    f"""
                                    <div style="background-color:#cff4fc;padding:20px;border-radius:10px;">
                                        <h4 style="color:#055160;">⏱️ Waktu Komputasi</h4>
                                        <h2 style="margin:0;color:#055160;">{elapsed_time:.2f} detik</h2>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                            col37, col38 = st.columns([0.5, 0.5])
                            with col37:
                                st.write("**Confusion Matrix:**")
                                conf_matrix = results['confusion_matrix']
                                labels = ['TIDAK ERAT', 'ERAT']  # Ubah sesuai label datamu

                                fig, ax = plt.subplots()
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                            xticklabels=labels, yticklabels=labels, cbar=True, ax=ax)
                                ax.set_xlabel('Predicted label')
                                ax.set_ylabel('True label')
                                ax.set_title('Confusion Matrix')
                                st.pyplot(fig)

                            with col38:
                                st.write("**Laporan Klasifikasi:**")
                                st.dataframe(results['classification_report'])
                                st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                            col39, col41 = st.columns([0.5, 0.5])
                            with col39:
                                st.write("**Precision:**", results['precision'])
                                st.write("**Recall:**", results['recall'])
                                st.write("**F1-Score:**", results['f1_score'])
                                st.write("**Specificity:**", results['specificity'])
                                st.write("**NPV:**", results['npv'])
                                # st.write("***Mean Accuracy:***", results['average_accuracy'])

                            with col41:
                                st.write("**Cohen's Kappa:**", results['kappa'])
                                st.write("**ROC AUC Score:**", results["roc_auc"])
                                # st.write("**Fold Terbaik:**", results['fold_terbaik'])
                                # st.write("***Mean Error:***", results['average_error'])
                                st.write("**Data Training:**", results['jumlah_data_training'])
                                st.write("**Data Testing:**", results['jumlah_data_testing'])

                            st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                            col391, col411 = st.columns([0.5, 0.5])
                            with col391:
                                st.image(results['n_tree_plot'], caption="Akurasi vs n-tree (n_estimators)")
                            with col411:
                                st.image(results['mtry_plot'], caption="Akurasi vs mtry (max_features)")
                            st.write("**Visualisasi Salah Satu Pohon dari Random Forest:**")
                            st.image(results['tree_image'], caption="Random Forest Tree", use_container_width=True)

                            st.write("**Feature Importance:**")
                            st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)

                            st.write("**Kurva ROC-AUC:**")
                            st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algoritma_select == "Support Vector Machine":

                    if st.button("Jalankan SVM"):
                        # Catat waktu mulai
                        start_time = time.time()
                        results = run_svm(df_final, n_splits=jmlFold)  # Disesuaikan dengan parameter baru
                        # Catat waktu selesai
                        end_time = time.time()
                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi SVM</h3>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"""
                                <div style="background-color:#d1e7dd;padding:20px;border-radius:10px;">
                                    <h4 style="color:#0f5132;">🎯 Akurasi Model</h4>
                                    <h2 style="margin:0;color:#0f5132;">{results['accuracy'] * 100:.2f}%</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        with col2:
                            st.markdown(
                                f"""
                                <div style="background-color:#cff4fc;padding:20px;border-radius:10px;">
                                    <h4 style="color:#055160;">⏱️ Waktu Komputasi</h4>
                                    <h2 style="margin:0;color:#055160;">{elapsed_time:.2f} detik</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                        col42, col43 = st.columns([0.5, 0.5])
                        with col42:
                            st.write("**Confusion Matrix:**")
                            conf_matrix = results['confusion_matrix']
                            labels = ['TIDAK ERAT', 'ERAT']  # Ubah sesuai label datamu

                            fig, ax = plt.subplots()
                            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=labels, yticklabels=labels, cbar=True, ax=ax)
                            ax.set_xlabel('Predicted label')
                            ax.set_ylabel('True label')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)

                        with col43:
                            st.write("**Laporan Klasifikasi:**")
                            st.dataframe(results['classification_report'])
                            st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                        col300, col400, col500 = st.columns([0.15,0.7, 0.15])
                        with col400:
                            st.write("**Akurasi setiap kernel:**")
                            st.dataframe(results['best_accuracy_per_kernel'])
                        col44, col45 = st.columns([0.5, 0.5])
                        with col44:
                            st.write("**Precision:**", results['precision'])
                            st.write("**Recall:**", results['recall'])
                            st.write("**F1-Score:**", results['f1_score'])
                            st.write("**Specificity:**", results['specificity'])
                            st.write("**NPV:**", results['npv'])
                            # st.write("***Mean Accuracy:***", results['average_accuracy'])

                        with col45:
                            st.write("**Cohen's Kappa:**", results['kappa'])
                            st.write("**ROC AUC Score:**", results["roc_auc"])
                            # st.write("**Fold Terbaik:**", results['fold_terbaik'])
                            # st.write("***Mean Error:***", results['average_error'])
                            # st.write("**Data Training:**", results['jumlah_data_training'])
                            # st.write("**Data Testing:**", results['jumlah_data_testing'])

                        st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)

                        st.write("**Feature Importance (dengan Permutation Importance):**")
                        st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)

                        # st.write("**Pengaruh Fitur:**")
                        # st.dataframe(results['feature_target_correlation'])
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)

                        with st.expander("Hyperparameter Terbaik"):
                            st.write(results['best_params'])




        elif fitur == "Sampling Data":
            
            def proses_baak(file):
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:
                            excel_file = pd.ExcelFile(file)
                            processed_sheets = []
                            for sheet in excel_file.sheet_names:
                                df = pd.read_excel(file, sheet_name=sheet, usecols="B,C,D,E", skiprows=1)  # Select attribute columns
                                df.columns = ["nim", "nama", "lama studi", "ipk"]  # Set column names
                                
                                df = df.dropna()  # Remove rows with missing values
                                df = df[~df['nim'].apply(lambda x: str(x)[4:6] in ['01', '02', '03'])]  # Filter rows based on nim
                                
                                # Convert "lama studi" from year format to month format
                                def convert_lama_studi(value):
                                    value = str(value)
                                    if '.' in value:
                                        bagian_depan, bagian_belakang = value.split('.')
                                        bulan = int(bagian_depan) * 12 + int(bagian_belakang)
                                    else:
                                        bulan = int(value) * 12
                                    return bulan

                                df["lama studi"] = df["lama studi"].apply(convert_lama_studi)  # Apply conversion to "lama studi"
                                
                                # Multiply "ipk" by 100
                                df["ipk"] = df["ipk"] * 100  # Scale "ipk" values
                                
                                # Filter based on "lama studi" and "ipk" values
                                df = df[(df["lama studi"] >= 42) & (df["lama studi"] <= 60)]  # Filter "lama studi"
                                df = df[(df["ipk"] >= 250) & (df["ipk"] <= 400)]  # Filter "ipk"

                                # Categorize "ipk" into bins
                                def categorize_ipk(ipk_value):
                                    if 250 <= ipk_value <= 300:
                                        return 'RENDAH'
                                    elif 301 <= ipk_value <= 350:
                                        return 'MEDIAN'
                                    elif 351 <= ipk_value <= 400:
                                        return 'TINGGI'
                                    else:
                                        return None

                                df['kategori_ipk'] = df['ipk'].apply(categorize_ipk)  # Add "kategori_ipk" column
                                
                                # Categorize "lama studi" into bins
                                def categorize_lama_studi(lama_studi_value):
                                    if 42 <= lama_studi_value <= 48:
                                        return 'TEPAT'
                                    elif 49 <= lama_studi_value <= 54:
                                        return 'CUKUP'
                                    elif 55 <= lama_studi_value <= 60:
                                        return 'LAMA'
                                    else:
                                        return None

                                df['kategori_lama_studi'] = df['lama studi'].apply(categorize_lama_studi)  # Add "kategori_lama_studi" column
                                
                                processed_sheets.append(df)
                            df = pd.concat(processed_sheets, ignore_index=True) if processed_sheets else pd.DataFrame()

                            # Normalize NIM
                            df['nim'] = df['nim'].astype(str).str.strip()
                            df = df[df['nim'].str.isdigit()]  # Filter rows where "nim" is numeric
                            df = df[df['nim'].str.len() == 9]  # Ensure "nim" is 9 digits long

                        return df[["nim", "kategori_ipk", "kategori_lama_studi"]]
            
            def sampling_pkpa(files, sampling_fraction):
                sampled_data_list = []  # Untuk menyimpan hasil sampling dari tiap file

                kolom_filter_index = {
                    "C": 2,    # Kode Progdi
                    "D": 3,    # NIM
                    "AC": 28,  # ketereratan
                    "AJ": 35,  # kerja tim
                    "AA": 26   # beasiswa
                }

                for i, file in enumerate(files):
                    if file.name.endswith('.csv'):
                        st.warning(f"File {file.name} diabaikan: hanya file Excel yang didukung.")
                        continue

                    excel_file = pd.ExcelFile(file)
                    if "Rekap" not in excel_file.sheet_names:
                        st.warning(f"Sheet 'Rekap' tidak ditemukan di file: {file.name}")
                        continue

                    raw_df = pd.read_excel(file, sheet_name="Rekap", header=None)

                    if i == 0:
                        header_df = raw_df.iloc[0]  # Baris pertama sebagai header

                    df = raw_df[1:]  # Buang baris pertama
                    df.columns = header_df  # Set header

                    try:
                        col_names = {k: header_df[v] for k, v in kolom_filter_index.items()}
                    except IndexError:
                        st.warning(f"File {file.name} tidak memiliki cukup kolom.")
                        continue

                    try:
                        df = df.dropna(subset=[col_names["C"], col_names["D"], col_names["AC"], col_names["AJ"], col_names["AA"]])
                        df = df[~df[col_names["C"]].astype(str).str.strip().isin(['01', '02', '03'])]  # Kode Progdi
                        df = df[df[col_names["D"]].astype(str).str.isdigit() & (df[col_names["D"]].astype(str).str.len() == 9)]  # NIM
                        df = df[~df[col_names["AA"]].astype(str).str.strip().isin(["0"])]  # beasiswa
                    except KeyError:
                        st.warning(f"Kolom penting tidak ditemukan dalam file: {file.name}")
                        continue

                    # Stratified Sampling berdasarkan 'ketereratan' agar proporsi kelas tetap terjaga
                    df['ketereratan'] = df[col_names["AC"]].astype(str).str.strip()  # Mengambil kolom ketereratan
                    sampled_df = df.groupby('ketereratan').apply(lambda x: x.sample(frac=sampling_fraction, random_state=42))
                    sampled_df = sampled_df.reset_index(drop=True)

                    sampled_data_list.append(sampled_df)

                # Menggabungkan hasil sampling dari semua file
                combined_sampled_data = pd.concat(sampled_data_list, ignore_index=True)

                return combined_sampled_data
            
            def filter_dan_encoding_sampel_pkpa(df):
                # Pastikan kolom yang diperlukan ada dalam dataframe
                required_columns = ["Kode Progdi", "nimhsmsmh", "nmmhsmsmh", "f5b", "f1201", "f14", "f1766"]
                if not all(col in df.columns for col in required_columns):
                    st.error("Kolom yang diperlukan tidak ada!")
                    return pd.DataFrame()

                # Pilih kolom yang relevan dari dataframe berdasarkan required_columns
                df = df[required_columns]

                # Rename kolom-kolom untuk kejelasan
                df = df.rename(columns={
                    "kode": "Kode Progdi",
                    "nimhsmsmh": "nim",
                    "nmmhsmsmh": "nama",
                    "f5b": "pekerjaan",
                    "f1201": "beasiswa",
                    "f14": "ketereratan",
                    "f1766": "kerja tim"
                })
                
                # Bersihkan data dari nilai kosong
                df = df.dropna()
                
            # Encoding untuk 'ketereratan'
                df['ketereratan'] = df['ketereratan'].astype(str).str.strip()
                df = df[df['ketereratan'].isin(['1', '2', '3', '4', '5'])]  # Pastikan hanya nilai yang valid
                df['ketereratan'] = df['ketereratan'].astype(int)
                
                # Filter dan encoding untuk 'beasiswa'
                df = df[~df['beasiswa'].astype(str).str.strip().isin(["0"])]  # Pastikan tidak ada nilai "0"
                df['beasiswa'] = df['beasiswa'].astype(str).str.strip()
                df = df[df['beasiswa'].isin(['1', '2', '3', '4', '5', '6', '7'])]
                df['beasiswa'] = df['beasiswa'].astype(int)

                # Filter dan encoding untuk 'kerja tim'
                df = df[~df['kerja tim'].astype(str).str.strip().isin(["0"])]  # Pastikan tidak ada nilai "0"
                df['kerja tim'] = df['kerja tim'].astype(str).str.strip()
                df = df[df['kerja tim'].isin(['1', '2', '3', '4', '5'])]
                df['kerja tim'] = df['kerja tim'].astype(int)
                
                # Mengubah 'ketereratan' menjadi kategori 'ERAT' atau 'TIDAK ERAT'
                df['ketereratan'] = df['ketereratan'].apply(lambda x: 0 if x in [1, 2, 3] else 1)
                df['beasiswa'] = df['beasiswa'].apply(lambda x: 0 if x in ['2', '3', '4', '5', '6', '7'] else 1)

                # Filter 'Kode Progdi' agar tidak termasuk '01', '02', '03'
                df = df[~df['Kode Progdi'].isin(['01', '02', '03'])]

                # Normalisasi NIM dan pastikan hanya angka dengan panjang 9 digit
                df['nim'] = df['nim'].astype(str).str.strip()
                df = df[df['nim'].str.isdigit()]
                df = df[df['nim'].str.len() == 9]
                
                # Encoding untuk kolom kategorikal seperti 'Kode Progdi', 'Nama', dsb.
                label_encoder = LabelEncoder()
                
                # Melakukan encoding pada kolom yang mengandung nilai kategorikal
                df['Kode Progdi'] = label_encoder.fit_transform(df['Kode Progdi'].astype(str))
                df['nim'] = label_encoder.fit_transform(df['nim'].astype(str))
                df['nama'] = label_encoder.fit_transform(df['nama'].astype(str))

                return df[["nim", "beasiswa", "ketereratan", "kerja tim"]]  # Kembalikan kolom yang relevan
            
            def evaluate_model(df):
                # Split data menjadi fitur dan target
                X = df.drop('ketereratan', axis=1)
                y = df['ketereratan']

                # Terapkan SMOTE untuk menyeimbangkan data
                X_resampled, y_resampled = apply_smote(X, y)

                # Split ke dalam training dan testing
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

                # Inisialisasi dan latih Decision Tree
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)

                # Prediksi dan hitung akurasi
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                return accuracy
            
            def play_success_sound():
                db = connect_to_db()
                cursor = db.cursor()
                user = st.session_state.user
                user_id = user["id_pengguna"]
                cursor.execute("SELECT nada_notif FROM pengguna WHERE id_pengguna=%s", (user_id,))
                result = cursor.fetchone()
                cursor.close()
                db.close()

                if result and result[0]:
                    encoded_audio = base64.b64encode(result[0]).decode()
                    st.markdown(f"""
                    <audio autoplay>
                        <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
                    </audio>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("🔇 Tidak ada nada notifikasi ditemukan untuk pengguna ini.")

            data_raw_pkpa = st.sidebar.file_uploader(
                "Upload data PKPA", 
                type=["xlsx", "xls"], 
                accept_multiple_files=True
            )
            data_raw_baak = st.sidebar.file_uploader(
                "Upload data BAAK", 
                type=["xlsx", "xls"], 
                accept_multiple_files=False
            )

            # Proses jika file diunggah
            if data_raw_pkpa and data_raw_baak:
                # sampling_fraction = random.uniform(0.20, 0.8)
                # # sampling_fractions = [0.23, 0.46, 0.15, 0.20, 0.43, 0.70, 0.80, 0.63, 0.53]
                st.sidebar.write("Masukkan proporsi sampling (pisahkan dengan koma)")
                sampling_input_min = st.sidebar.text_input(
                    "MIN %",key="input_min",
                    placeholder="Contoh: 0.25",
                    help="Masukkan satu atau beberapa nilai proporsi sampling (0 - 1)"
                )
                sampling_input_max = st.sidebar.text_input(
                    "MAX %",key="input_max",
                    placeholder="Contoh: 0.85",
                    help="Masukkan satu atau beberapa nilai proporsi sampling (0 - 1)"
                )
                input_target_akurasi = st.sidebar.text_input(
                    "Akurasi Diharapkan",key="target_akurasi",
                    placeholder="Contoh: 0.75",
                )
                if sampling_input_min and sampling_input_max and input_target_akurasi:
                    try:
                        # Konversi ke list float
                        target_akurasi = float(input_target_akurasi)
                        min_vals = [float(x.strip()) for x in sampling_input_min.split(",") if x.strip()]
                        max_vals = [float(x.strip()) for x in sampling_input_max.split(",") if x.strip()]

                        # Ambil nilai min dan max
                        min_range = min(min_vals)
                        max_range = max(max_vals)

                        if min_range >= max_range:
                            st.sidebar.error("Nilai minimum harus lebih kecil dari maksimum.")
                        else:
                            if st.sidebar.button("Buat Sampling"):
                                with st.spinner("Mencari proporsi sampling terbaik..."):
                                    target_fraction = None  # Untuk menyimpan nilai sampling_fraction yang berhasil

                                    # Sampling awal untuk header dan cleaning
                                    combined_sampled_data = sampling_pkpa(data_raw_pkpa, 0.1)
                                    processed_data = filter_dan_encoding_sampel_pkpa(combined_sampled_data)
                                    baak_data = proses_baak(data_raw_baak)

                                    if not processed_data.empty and not baak_data.empty:
                                        st.subheader("Data Hasil Sampling PKPA")
                                        processed_data['nim'] = processed_data['nim'].astype(str)
                                        baak_data['nim'] = baak_data['nim'].astype(str)

                                        # Gabungkan data PKPA dan BAAK berdasarkan NIM
                                        combined_data = pd.merge(processed_data, baak_data, on='nim', how='inner')

                                        # Konversi semua kolom ke string
                                        for col in combined_data.columns:
                                            combined_data[col] = combined_data[col].astype(str)

                                        accuracy = 0
                                        iteration = 0
                                        iterasi_placeholder = st.empty()

                                        while accuracy < target_akurasi:  # target_akurasi_float sudah dikonversi sebelumnya
                                            iteration += 1
                                            sampling_fraction = random.uniform(min_range, max_range)

                                            iterasi_placeholder.empty()
                                            iterasi_placeholder.markdown(
                                                f"<div style='font-size: 22px; font-weight: bold;'>Iterasi: {iteration}, Sampling Fraction: {sampling_fraction:.2f}, Akurasi: {accuracy*100:.2f}%</div>",
                                                unsafe_allow_html=True
                                            )

                                            sampled_data = sampling_pkpa(data_raw_pkpa, sampling_fraction)
                                            processed_data = filter_dan_encoding_sampel_pkpa(sampled_data)

                                            if processed_data.empty:
                                                continue  # Lewati jika hasil kosong

                                            accuracy = evaluate_model(processed_data)

                                            if accuracy >= target_akurasi:
                                                target_fraction = sampling_fraction  # Simpan nilai sampling_fraction yang berhasil
                                                iterasi_placeholder.markdown(
                                                    f"<div style='font-size: 22px; font-weight: bold;'>Iterasi: {iteration}, Akurasi: {accuracy*100:.2f}% ✅</div>",
                                                    unsafe_allow_html=True
                                                )
                                                st.success(f"Akurasi tercapai! Proporsi sampling terbaik: **{sampling_fraction:.2f}**")

                                                # Tombol unduhan
                                                buffer_xlsx = io.BytesIO()
                                                with pd.ExcelWriter(buffer_xlsx, engine='xlsxwriter') as writer:
                                                    sampled_data.to_excel(writer, sheet_name='Rekap', index=False)
                                                buffer_xlsx.seek(0)
                                                play_success_sound()
                                                st.download_button("Unduh Data Sample PKPA", data=buffer_xlsx,
                                                                file_name="Sampling_PKPA.xlsx",
                                                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                                                break
                    except ValueError:
                        st.sidebar.error("Input harus berupa angka desimal, dipisahkan dengan koma.")  
                else:
                    st.markdown("<div style='padding-top: 20rem;'>", unsafe_allow_html=True)
                    st.warning("Tentukan proporsi sampling file PKPA yang diinginkan")
            else:
                st.markdown("<div style='padding-top: 20rem;'>", unsafe_allow_html=True)
                st.info("Silakan upload terlebih dahulu data BAAK dan PKPA yang ingin dilakukan sampling pada sidebar")