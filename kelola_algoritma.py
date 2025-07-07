import streamlit as st
import mysql.connector
import pandas as pd
import time
from datetime import datetime

def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="data_mining_system"
    )

def update_algoritma(kode, nama_baru):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("UPDATE algoritma SET nama_algoritma = %s WHERE kode_algoritma = %s", (nama_baru, kode))
    db.commit()
    cursor.close()
    db.close()

def delete_algoritma(kode):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM algoritma WHERE kode_algoritma = %s", (kode,))
    db.commit()
    cursor.close()
    db.close()

def main():
    st.subheader("Kelola Algoritma")
    st.markdown("Berikut data algoritma yang tersimpan di sistem")

    # Tombol pencarian dan tambah
    col1, col2, col3, col4 = st.columns([6, 5, 3, 0.75])
    with col1:
        search_query = st.text_input("🔍 Cari Nama Algoritma", placeholder="Cari nama algoritma...", label_visibility="hidden")
    with col3:
        st.markdown("<div style='padding-top: 1.8rem;'>", unsafe_allow_html=True)
        if st.button("Tambah Algoritma Baru"):
            st.session_state.hal_kelola_algoritma = "tambah_algoritma"
            st.rerun()

    st.markdown("---")

    # Ambil data
    db = connect_to_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT kode_algoritma, nama_algoritma, tanggal_dibuat FROM algoritma ORDER BY tanggal_dibuat DESC")
    data = cursor.fetchall()
    cursor.close()
    db.close()

    # Filter pencarian
    if search_query:
        data = [alg for alg in data if search_query.lower() in alg['nama_algoritma'].lower()]

    # State untuk aksi
    if "edit_kode" not in st.session_state:
        st.session_state.edit_kode = None
    if "delete_confirm" not in st.session_state:
        st.session_state.delete_confirm = None
    if "edit_confirm" not in st.session_state:
        st.session_state.edit_confirm = None

    if data:
        colA, colB, colC, colD, colE = st.columns([2, 5, 3, 1, 1])
        with colA:
            st.write("**Kode Algoritma**")
        with colB:
            st.write("**Nama Algoritma**")
        with colC:
            st.write("**Tanggal Dibuat**")
        with colD:
            st.write("")
        with colE:
            st.write("")
            
        st.markdown("<div style='padding-top: 1.2rem;'>", unsafe_allow_html=True)
        for alg in data:
            kode_algoritma = alg["kode_algoritma"]
            nama_algoritma = alg["nama_algoritma"]
            tanggal_dibuat = alg["tanggal_dibuat"]

            # mode edit
            if st.session_state.edit_kode == kode_algoritma:

                colA, colB, colC, colD, colE = st.columns([2, 5, 3, 1, 1])
                with colA:
                    st.markdown(f"<span style='color: orange; font-weight: bold;'>{kode_algoritma}</span>", unsafe_allow_html=True)
                with colB:
                    new_nama = st.text_input("", value=nama_algoritma, key=f"edit_input_{kode_algoritma}",label_visibility="collapsed")
                with colC:
                    st.write(pd.to_datetime(tanggal_dibuat).strftime("%d %B %Y %H:%M"))
                with colD:
                    if st.button("💾"):
                        if new_nama.strip() == nama_algoritma.strip():
                            st.info("Tidak ada perubahan data.")
                            st.session_state.edit_kode = None
                            st.rerun()
                        else:
                            st.session_state.edit_confirm = (kode_algoritma, new_nama)
                            st.rerun()
                with colE:
                    if st.button("❌"):
                        st.session_state.edit_kode = None
                        st.rerun()

            else:
                # Tabel tampil normal
                col1, col2, col3, col4, col5 = st.columns([2, 5, 3, 1, 1])
                with col1:
                    st.write(kode_algoritma)
                with col2:
                    st.write(nama_algoritma)
                with col3:
                    st.write(pd.to_datetime(alg["tanggal_dibuat"]).strftime("%d %B %Y %H:%M"))
                with col4:
                    if st.button("✏️", key=f"edit_{kode_algoritma}"):
                        st.session_state.edit_kode = kode_algoritma
                        st.rerun()
                with col5:
                    if st.button("🗑️", key=f"delete_{kode_algoritma}"):
                        st.session_state.delete_confirm = kode_algoritma
                        st.rerun()

        st.caption(f"Total algoritma: **{len(data)}**")

    else:
        st.info("📭 Belum ada data algoritma atau tidak ditemukan hasil pencarian.")

    st.markdown("---")

    # Pop-up konfirmasi hapus
    if st.session_state.delete_confirm:
        kode = st.session_state.delete_confirm
        st.warning(f"⚠️ Apakah Anda yakin ingin menghapus algoritma dengan kode **{kode}**?")
        colC1, colC2 = st.columns(2)
        with colC1:
            if st.button("✅ Ya, Hapus"):
                delete_algoritma(kode)
                st.success("🗑️ Algoritma berhasil dihapus.")
                st.session_state.delete_confirm = None
                time.sleep(1)
                st.rerun()
        with colC2:
            if st.button("❌ Batal Hapus"):
                st.session_state.delete_confirm = None
                st.rerun()

    # Pop-up konfirmasi simpan edit
    if st.session_state.edit_confirm:
        kode, nama_baru = st.session_state.edit_confirm
        st.warning(f"⚠️ Yakin ingin mengubah nama algoritma **{kode}** menjadi:\n\n➡️ **{nama_baru}** ?")
        colS1, colS2 = st.columns(2)
        with colS1:
            if st.button("✅ Ya, Simpan Perubahan"):
                update_algoritma(kode, nama_baru)
                st.success("✅ Nama algoritma berhasil diperbarui.")
                st.session_state.edit_kode = None
                st.session_state.edit_confirm = None
                time.sleep(1)
                st.rerun()
        with colS2:
            if st.button("❌ Batal Edit"):
                st.session_state.edit_confirm = None
                st.rerun()
