# Import Library
import streamlit as st
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import io
import time
import uuid
import base64
import mysql.connector
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc,cohen_kappa_score)
from sklearn.tree import plot_tree
from datetime import datetime

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="data_mining_system"
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
        encoded_audio = base64.b64encode(result[0]).decode()
        st.markdown(f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)
    else:
        st.warning("🔇 Tidak ada nada notifikasi ditemukan untuk pengguna ini.")
def generate_kode_model(prefix="ML"):
    # Ambil 6 karakter pertama dari UUID
    unique_id = uuid.uuid4().hex[:6].upper()
    return f"{prefix}.{unique_id}"

def serialize_object(obj):
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()

def save_model_to_db(
    model_object,
    kode_model,
    kode_algoritma,
    jumlah_fold,
    akurasi_model,
    id_pengguna,
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

        # Cek jika sudah ada kombinasi kode_algoritma + id_pengguna
        check_sql = """
            SELECT kode_model FROM models 
            WHERE kode_algoritma = %s AND id_pengguna = %s
        """
        cursor.execute(check_sql, (kode_algoritma, id_pengguna))
        existing = cursor.fetchone()

        # Jika ada, arsipkan model lama ke arsip_models dan hapus dari models
        if existing:
            existing_kode_model = existing[0]

            # Arsipkan ke arsip_models
            archive_sql = """
                INSERT INTO arsip_models (
                    kode_model, kode_algoritma, file_model, jumlah_fold, 
                    akurasi, tanggal_dibuat, tanggal_dihapus, id_pengguna, nama_pengguna
                )
                SELECT 
                    kode_model, kode_algoritma, file_model, jumlah_fold, 
                    akurasi, tanggal_dibuat, NOW(), id_pengguna, nama_pengguna
                FROM models
                WHERE kode_model = %s
            """
            cursor.execute(archive_sql, (existing_kode_model,))

            # Hapus model lama
            delete_sql = "DELETE FROM models WHERE kode_model = %s"
            cursor.execute(delete_sql, (existing_kode_model,))

        # Serialisasi model baru
        model_binary = serialize_object(model_object)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ambil nama_pengguna dari session
        nama_pengguna = st.session_state.user["nama_pengguna"]

        # Simpan model baru
        insert_sql = """
            INSERT INTO models (
                kode_model, kode_algoritma, file_model, jumlah_fold, 
                akurasi, tanggal_dibuat, id_pengguna, nama_pengguna
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            kode_model,
            kode_algoritma,
            model_binary,
            jumlah_fold,
            round(akurasi_model, 4),
            now,
            id_pengguna,
            nama_pengguna
        )
        cursor.execute(insert_sql, values)
        conn.commit()

        print(f"✅ Model '{kode_model}' berhasil disimpan ke database.")

    except mysql.connector.Error as err:
        print(f"❌ Gagal menyimpan model ke database: {err}")

    finally:
        cursor.close()
        conn.close()

def save_encoder_to_db(encoders_dict, host="localhost", user="root", password="", database="data_mining_system"):
    """
    encoders_dict: dict dengan format {kode_encoder: (nama_encoder, encoder_object)}
    """
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()

        for kode_encoder, (nama_encoder, encoder_object) in encoders_dict.items():
            encoder_binary = serialize_object(encoder_object)

            delete_sql = "DELETE FROM encoder WHERE kode_encoder = %s"
            cursor.execute(delete_sql, (kode_encoder,))

            insert_sql = """
                INSERT INTO encoder (kode_encoder, nama_encoder, file_encoder, tanggal_dibuat)
                VALUES (%s, %s, %s, %s)
            """
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(insert_sql, (kode_encoder, nama_encoder, encoder_binary, now))

            print(f"✅ Encoder '{nama_encoder}' disimpan sebagai '{kode_encoder}'.")

        conn.commit()

    except mysql.connector.Error as err:
        print(f"❌ Gagal menyimpan encoder ke database: {err}")

    finally:
        cursor.close()
        conn.close()

# Fungsi utama
def run_random_forest(df_final, binary_class=True, n_splits=10):
    with st.spinner('Harap tunggu, sedang memproses prediksi...'):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Menyiapkan data dan melakukan Label Encoding...")
        time.sleep(1)
        progress_bar.progress(5)

        data = df_final[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa', 'ketereratan']].copy()

        le_ipk = LabelEncoder()
        le_studi = LabelEncoder()
        le_kerjatim = LabelEncoder()
        le_beasiswa = LabelEncoder()
        le_ketereratan = LabelEncoder()

        data['kategori_ipk'] = le_ipk.fit_transform(data['kategori_ipk'].astype(str))
        data['kategori_lama_studi'] = le_studi.fit_transform(data['kategori_lama_studi'].astype(str))
        data['kerja tim'] = le_kerjatim.fit_transform(data['kerja tim'].astype(str))
        data['beasiswa'] = le_beasiswa.fit_transform(data['beasiswa'].astype(str))
        data['ketereratan'] = le_ketereratan.fit_transform(data['ketereratan'].astype(str))

        progress_bar.progress(15)

        # === Tahap 2: Setup data dan Stratified K-Fold ===
        status_text.text("📊 Menyiapkan data training/testing dengan StratifiedKFold...")
        X = data[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa']].values
        y = data['ketereratan'].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        progress_bar.progress(20)

        all_y_test = []
        all_y_pred = []
        all_y_proba = []
        results_by_params = []

        status_text.text("🧠 Melatih model Random Forest dengan GridSearchCV...")

        total_folds = skf.get_n_splits()
        for i, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
            status_text.text(f"🔁 Fold {i}/{total_folds} sedang diproses...")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            param_grid = {
                'n_estimators': [50, 100, 150, 200, 250, 300, 350],
                'max_features': [1, 2, 3, 4, 5, 6, 7],
                'max_depth': [None],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'criterion': ['gini']
            }

            grid_search = GridSearchCV(RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True),
                                    param_grid, cv=3, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            mean_scores = grid_search.cv_results_['mean_test_score']
            params_list = grid_search.cv_results_['params']
            for p, acc in zip(params_list, mean_scores):
                results_by_params.append({
                    'n_estimators': p['n_estimators'],
                    'max_features': p['max_features'],
                    'accuracy': acc
                })

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)

            progress = 20 + int((50 * i) / total_folds)
            progress_bar.progress(progress)

        all_y_test = np.array(all_y_test)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)

        status_text.text("🧾 Menyimpan hasil prediksi dan menghitung metrik evaluasi...")
        hasil_prediksi_df = pd.DataFrame({
            'Actual': all_y_test,
            'Predicted': all_y_pred
        })
        hasil_prediksi_df.to_excel('data_pred/data_prediksi_RF.xlsx', index=False)

        accuracy = accuracy_score(all_y_test, all_y_pred)
        report_df = pd.DataFrame(classification_report(all_y_test, all_y_pred, target_names=le_ketereratan.classes_, output_dict=True)).transpose()
        conf_matrix = confusion_matrix(all_y_test, all_y_pred)
        kappa = cohen_kappa_score(all_y_test, all_y_pred)

        if binary_class:
            precision = precision_score(all_y_test, all_y_pred)
            recall = recall_score(all_y_test, all_y_pred)
            f1 = f1_score(all_y_test, all_y_pred)
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            npv = tn / (tn + fn) if (tn + fn) != 0 else 0
            roc_auc = roc_auc_score(all_y_test, all_y_proba[:, 1])
        else:
            precision = precision_score(all_y_test, all_y_pred, average='weighted')
            recall = recall_score(all_y_test, all_y_pred, average='weighted')
            f1 = f1_score(all_y_test, all_y_pred, average='weighted')
            roc_auc = roc_auc_score(all_y_test, all_y_proba, multi_class='ovr')
            specificity = None
            npv = None

        progress_bar.progress(80)
        status_text.text("📉 Membuat visualisasi evaluasi model...")

        results_df = pd.DataFrame(results_by_params)

        # Accuracy vs n_estimators
        plt.figure(figsize=(6, 4))
        mean_n = results_df.groupby('n_estimators')['accuracy'].mean()
        plt.plot(mean_n.index, mean_n.values, marker='o', color='darkorange')
        plt.xlabel("n-tree (n_estimators)")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs n-tree")
        plt.grid(True)
        plt.tight_layout()
        n_tree_plot_path = "accuracy_vs_n_tree.png"
        plt.savefig(n_tree_plot_path)
        plt.close()

        # Accuracy vs max_features
        plt.figure(figsize=(6, 4))
        mean_m = results_df.groupby('max_features')['accuracy'].mean()
        plt.plot(mean_m.index, mean_m.values, marker='o', color='steelblue')
        plt.xlabel("mtry (max_features)")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs mtry")
        plt.grid(True)
        plt.tight_layout()
        mtry_plot_path = "accuracy_vs_mtry.png"
        plt.savefig(mtry_plot_path)
        plt.close()

        # Feature Importance
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Fitur': ['kategori_ipk', 'kategori_lama_studi', 'kerja tim','beasiswa'],
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Fitur'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance Random Forest')
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # Visualisasi pohon
        tree_fig_path = "tree.png"
        plt.figure(figsize=(20, 10))
        plot_tree(model.estimators_[0],
                feature_names=['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa'],
                class_names=le_ketereratan.classes_,
                filled=True, rounded=True, fontsize=8)
        plt.savefig(tree_fig_path)
        plt.close()

        # ROC Curve
        plt.figure(figsize=(8, 6))
        if binary_class:
            fpr, tpr, _ = roc_curve(all_y_test, all_y_proba[:, 1])
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        else:
            y_test_binarized = label_binarize(all_y_test, classes=np.unique(y))
            for i, color in zip(range(len(le_ketereratan.classes_)), ['blue', 'red', 'green', 'orange', 'purple']):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], all_y_proba[:, i])
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, label=f'Kelas {le_ketereratan.classes_[i]} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC-AUC Random Forest')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve.png")
        plt.close()

        progress_bar.progress(95)
        status_text.text("💾 Menyimpan model dan encoder...")

        results = {
            'accuracy': accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'kappa': kappa,
            'specificity': specificity,
            'npv': npv,
            'feature_importance': feature_importance_df,
            'feature_importance_image': 'feature_importance.png',
            'roc_auc': roc_auc,
            'roc_image': 'roc_curve.png',
            'tree_image': tree_fig_path,
            'n_tree_plot': n_tree_plot_path,
            'mtry_plot': mtry_plot_path,
            'y_test_labels': le_ketereratan.inverse_transform(all_y_test),
            'y_pred_labels': le_ketereratan.inverse_transform(all_y_pred),
            'label_encoder': le_ketereratan,
            'jumlah_data_training': len(X) * (n_splits - 1) // n_splits,
            'jumlah_data_testing': len(X) // n_splits,
        }

        save_model_to_db(
            model_object=model,
            kode_model=generate_kode_model(),
            kode_algoritma='RF',
            jumlah_fold=n_splits,
            akurasi_model=accuracy,
            id_pengguna=st.session_state.get("id_pengguna")
        )
        
        encoders_to_save = {
            'ENC001': ('le_ipk', le_ipk),
            'ENC002': ('le_studi', le_studi),
            'ENC003': ('le_beasiswa', le_beasiswa),
            'ENC004': ('le_kerjatim', le_kerjatim),
            'ENC005': ('le_ketereratan', le_ketereratan),
        }
        save_encoder_to_db(encoders_to_save)

        progress_bar.progress(100)
        status_text.text("✅ Proses selesai!")
        play_success_sound()
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        print("Model Random Forest telah disimpan.")


    return results

