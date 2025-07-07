# decision_tree_model.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import io
import joblib
import uuid
import base64
import mysql.connector
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve,cohen_kappa_score, auc)
from sklearn.preprocessing import LabelEncoder
# from imblearn.combine import SMOTETomek
from datetime import datetime

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
    host="sql12.freesqldatabase.com",
    user="sql12788718", 
    password="zh24wGnPJN", 
    database="sql12788718" 
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


def save_encoder_to_db(encoders_dict,host="sql12.freesqldatabase.com",user="sql12788718",password="zh24wGnPJN",database="sql12788718"):
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


def run_decision_tree(df_final, n_splits=10):
    os.makedirs('plots', exist_ok=True)

    with st.spinner('Harap tunggu, sedang memproses prediksi...'):
        progress_bar = st.empty()
        status_text = st.empty()
        progress_bar.progress(2)
        time.sleep(1)
        status_text.text("🔄 Menyiapkan data dan melakukan Label Encoding...")
        progress_bar.progress(5)

        # Preprocessing
        data_uji = df_final[['kategori_ipk', 'kategori_lama_studi', 'beasiswa', 'kerja tim', 'ketereratan']].copy()
        le_ipk, le_studi, le_beasiswa = LabelEncoder(), LabelEncoder(), LabelEncoder()
        le_kerjatim, le_ketereratan = LabelEncoder(), LabelEncoder()
        progress_bar.progress(10)
        data_uji['kategori_ipk'] = le_ipk.fit_transform(data_uji['kategori_ipk'])
        data_uji['kategori_lama_studi'] = le_studi.fit_transform(data_uji['kategori_lama_studi'])
        data_uji['beasiswa'] = le_beasiswa.fit_transform(data_uji['beasiswa'])
        data_uji['kerja tim'] = le_kerjatim.fit_transform(data_uji['kerja tim'])
        data_uji['ketereratan'] = le_ketereratan.fit_transform(data_uji['ketereratan'])

        progress_bar.progress(15)

        X = data_uji[['kategori_ipk', 'kategori_lama_studi', 'beasiswa', 'kerja tim']].values
        y = data_uji['ketereratan'].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        progress_bar.progress(20)

        param_grid = {
            'criterion': ['entropy'],
            'max_depth': [3, 5, 7, 10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        }

        fold_errors = []
        max_depth_acc = {}
        min_samples_split_acc = {}
        min_samples_leaf_acc = {}
        all_y_test = []
        all_y_pred = []
        all_y_proba = []

        status_text.text("🧠 Melatih model Decision Tree dengan GridSearchCV...")
        total_folds = skf.get_n_splits()
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            status_text.text(f"🔁 Fold {i}/{total_folds} sedang diproses...")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            grid_search = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy'
            )
            grid_search.fit(X_train, y_train)

            results = grid_search.cv_results_
            for j in range(len(results['params'])):
                params = results['params'][j]
                mean_acc = results['mean_test_score'][j]
                max_d = params['max_depth']
                min_split = params['min_samples_split']
                min_leaf = params['min_samples_leaf']

                max_depth_acc[max_d] = max_depth_acc.get(max_d, []) + [mean_acc]
                min_samples_split_acc[min_split] = min_samples_split_acc.get(min_split, []) + [mean_acc]
                min_samples_leaf_acc[min_leaf] = min_samples_leaf_acc.get(min_leaf, []) + [mean_acc]

            model = grid_search.best_estimator_

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            error = 1 - accuracy
            fold_errors.append(error)

            best_params = grid_search.best_params_
            max_depth_val = best_params['max_depth']
            min_samples_split_val = best_params['min_samples_split']
            min_samples_leaf_val = best_params['min_samples_leaf']

            max_depth_acc[max_depth_val] = max_depth_acc.get(max_depth_val, []) + [accuracy]
            min_samples_split_acc[min_samples_split_val] = min_samples_split_acc.get(min_samples_split_val, []) + [accuracy]
            min_samples_leaf_acc[min_samples_leaf_val] = min_samples_leaf_acc.get(min_samples_leaf_val, []) + [accuracy]

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)

            progress = 20 + int((40 * i) / total_folds)
            progress_bar.progress(progress)

        # Evaluasi dan visualisasi

        avg_error = np.mean(fold_errors)
        all_y_test = np.array(all_y_test)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.array(all_y_proba)
        y_pred_labels = le_ketereratan.inverse_transform(all_y_pred)
        y_test_labels = le_ketereratan.inverse_transform(all_y_test)

        status_text.text("🧾 Menyimpan hasil prediksi dan menghitung metrik evaluasi...")
        hasil_prediksi_df = pd.DataFrame({
            'Actual': y_test_labels,
            'Predicted': y_pred_labels
        })
        hasil_prediksi_df.to_excel('data_pred/data_prediksi_DT.xlsx', index=False)
        progress_bar.progress(65)
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        report_df = pd.DataFrame(classification_report(y_test_labels, y_pred_labels, output_dict=True)).transpose()
        conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

        if conf_matrix.shape == (2, 2):
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            npv = tn / (tn + fn) if (tn + fn) != 0 else 0
        else:
            specificity = 0
            npv = 0

        recall = recall_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        precision = precision_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_test_labels, y_pred_labels)
        f1 = f1_score(y_test_labels, y_pred_labels, average='weighted', zero_division=0)

        progress_bar.progress(75)
        status_text.text("📉 Membuat visualisasi evaluasi model...")

        roc_auc = roc_auc_score(all_y_test, all_y_proba[:, 1])
        fpr, tpr, _ = roc_curve(all_y_test, all_y_proba[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC-AUC Decision Tree')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()

        progress_bar.progress(80)

        final_model = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy'
        )
        final_model.fit(X, y)
        model = final_model.best_estimator_

        importance = model.feature_importances_
        feature_names = ['kategori_ipk', 'kategori_lama_studi', 'beasiswa', 'kerja tim']
        feature_importance_df = pd.DataFrame({
            'Fitur': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Fitur'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        feature_importance_image = 'plots/feature_importance.png'
        plt.savefig(feature_importance_image)
        plt.close()

        progress_bar.progress(85)

        plt.figure(figsize=(16, 8))
        plot_tree(model, feature_names=feature_names, class_names=le_ketereratan.classes_, filled=True)
        plt.tight_layout()
        tree_image_path = 'plots/decision_tree.png'
        plt.savefig(tree_image_path)
        plt.close()

        avg_max_depth_acc = {k: np.mean(v) for k, v in max_depth_acc.items()}
        plt.figure()
        plt.plot(list(avg_max_depth_acc.keys()), list(avg_max_depth_acc.values()), marker='o', color='green')
        plt.title("Accuracy vs max_depth (Decision Tree)")
        plt.xlabel("max_depth")
        plt.ylabel("Accuracy")
        plt.grid(True)
        max_depth_plot_path = "plots/accuracy_vs_max_depth.png"
        plt.tight_layout()
        plt.savefig(max_depth_plot_path)
        plt.close()

        progress_bar.progress(90)

        avg_min_samples_split_acc = {k: np.mean(v) for k, v in min_samples_split_acc.items()}
        plt.figure()
        plt.plot(list(avg_min_samples_split_acc.keys()), list(avg_min_samples_split_acc.values()), marker='o', color='purple')
        plt.title("Accuracy vs min_samples_split (Decision Tree)")
        plt.xlabel("min_samples_split")
        plt.ylabel("Accuracy")
        plt.grid(True)
        min_split_plot_path = "plots/accuracy_vs_min_samples_split.png"
        plt.tight_layout()
        plt.savefig(min_split_plot_path)
        plt.close()

        avg_min_samples_leaf_acc = {k: np.mean(v) for k, v in min_samples_leaf_acc.items()}
        plt.figure()
        plt.plot(list(avg_min_samples_leaf_acc.keys()), list(avg_min_samples_leaf_acc.values()), marker='o', color='orange')
        plt.title("Accuracy vs min_samples_leaf (Decision Tree)")
        plt.xlabel("min_samples_leaf")
        plt.ylabel("Accuracy")
        plt.grid(True)
        min_leaf_plot_path = "plots/accuracy_vs_min_samples_leaf.png"
        plt.tight_layout()
        plt.savefig(min_leaf_plot_path)
        plt.close()

        progress_bar.progress(95)
        status_text.text("💾 Menyimpan model dan encoder...")

        results = {
            'accuracy': accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'npv': npv,
            'kappa': kappa,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'roc_image': 'roc_curve.png',
            'feature_importance': feature_importance_df,
            'feature_importance_image': feature_importance_image,
            'tree_image': tree_image_path,
            'max_depth_plot': max_depth_plot_path,
            'min_samples_split_plot': min_split_plot_path,
            'min_samples_leaf_plot': min_leaf_plot_path,
            'y_pred_labels': y_pred_labels,
            'y_actual_labels': y_test_labels,
            'average_error': avg_error,
            'average_accuracy': 1 - avg_error,
            'jumlah_data': len(X)
        }
        save_model_to_db(
            model_object=model,
            kode_model=generate_kode_model(),
            kode_algoritma='DT',
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
        print("Model Decision Tree telah disimpan.")

        

    return results



