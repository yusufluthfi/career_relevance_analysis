import time
import streamlit as st
from decision_tree import (
    save_model_to_db,
    save_encoder_to_db,
    generate_kode_model,
    connect_db
)

def handle_simpan_model(results, jmlFold, set_kode_algoritma):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Apakah Anda ingin menyimpan model ini ?")

    colX, colY, _ = st.columns([2, 2, 9])
    with colX:
        simpan_model = st.button("Ya")
    with colY:
        tidak_simpan = st.button("Tidak")

    if simpan_model:
        id_pengguna = st.session_state.get("id_pengguna")
        kode_algoritma = set_kode_algoritma

        db = connect_db()
        cursor = db.cursor()
        cursor.execute("""
            SELECT kode_model, 
                   (SELECT nama_algoritma FROM algoritma WHERE kode_algoritma = models.kode_algoritma), 
                   akurasi 
            FROM models 
            WHERE kode_algoritma = %s AND id_pengguna = %s
        """, (kode_algoritma, id_pengguna))
        existing = cursor.fetchone()
        cursor.close()
        db.close()

        if existing:
            kode_model_lama, nama_algoritma_lama, akurasi_lama = existing
            st.warning(f"""
            🔁 Sudah ada model dengan algoritma **`{nama_algoritma_lama}`** milik Anda.

            Tindakan ini akan **menghapus model `{kode_model_lama}`** 
            dengan akurasi **{akurasi_lama * 100:.2f}%**
            dan menggantinya dengan model baru.
            """, icon="⚠️")

            colA, colB = st.columns(2)
            with colA:
                if st.button("Tetap Simpan"):
                    _simpan_model(results, jmlFold, kode_algoritma, id_pengguna)
            with colB:
                if st.button("Tidak Menyimpan"):
                    st.info("✅ Terima kasih telah konfirmasi. Model tidak disimpan.")
                    time.sleep(2)
                    st.rerun()
        else:
            _simpan_model(results, jmlFold, kode_algoritma, id_pengguna)

    if tidak_simpan:
        st.info("✅ Terima kasih telah konfirmasi. Model tidak disimpan.")
        time.sleep(2)
        st.rerun()


def _simpan_model(results, jmlFold, kode_algoritma, id_pengguna):
    kode_model_baru = generate_kode_model()
    save_model_to_db(
        model_object=results['model'],
        kode_model=kode_model_baru,
        kode_algoritma=kode_algoritma,
        jumlah_fold=jmlFold,
        akurasi_model=results['accuracy'],
        id_pengguna=id_pengguna
    )
    encoders_to_save = {
        'ENC001': ('le_ipk', results['encoders']['le_ipk']),
        'ENC002': ('le_studi', results['encoders']['le_studi']),
        'ENC003': ('le_beasiswa', results['encoders']['le_beasiswa']),
        'ENC004': ('le_kerjatim', results['encoders']['le_kerjatim']),
        'ENC005': ('le_ketereratan', results['encoders']['le_ketereratan']),
    }
    save_encoder_to_db(encoders_to_save)

    st.success(f"✅ Model berhasil disimpan sebagai **{kode_model_baru}**.")
    kode_algoritma = None
    time.sleep(2)
    st.rerun()
