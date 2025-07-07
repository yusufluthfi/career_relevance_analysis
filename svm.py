from sklearn.calibration import label_binarize
import streamlit as st 
import pandas as pd
import numpy as np
import os
import joblib
import io
import time
import shap
import uuid
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc, cohen_kappa_score
from sklearn.inspection import permutation_importance

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

def calculate_gamma_value(gamma, X_train):
    if gamma == 'scale':
        return 1 / (X_train.shape[1] * X_train.var())
    elif gamma == 'auto':
        return 1 / X_train.shape[1]
    else:
        return gamma  # for numeric gamma values

def run_svm(df_final, binary_class=True, n_splits=10):

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
        status_text.text("📊 Menyiapkan data training/testing dengan StratifiedKFold...")

        X = data[['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa']].values
        y = data['ketereratan'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        crossvalidation = StratifiedKFold(n_splits, shuffle=True, random_state=42)

        progress_bar.progress(20)
        
        all_results = []
        y_tests = []
        y_preds = []
        y_probas = []
        all_fold_errors = []
        all_conf_matrices = []
        feature_importances = []

        status_text.text("🧠 Melatih model SVM dengan GridSearchCV...")
        total_folds = crossvalidation.get_n_splits()
        for i, (train_index, test_index) in enumerate(crossvalidation.split(X_scaled, y)):
            status_text.text(f"🔁 Fold {i}/{total_folds} sedang diproses...")
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]

            param_grid = {
                'kernel': ['linear', 'poly', 'rbf'],
                'C': [0.01, 0.1, 1, 10.0],
                'gamma': ['scale'],
            }

            grid_search = GridSearchCV(SVC(probability=True, random_state=42),
                                       param_grid, cv=3, scoring='accuracy', n_jobs=-1, return_train_score=True)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            for j in range(len(grid_search.cv_results_['params'])):
                result = grid_search.cv_results_['params'][j].copy()
                result['mean_test_accuracy'] = grid_search.cv_results_['mean_test_score'][j]
                result['fold'] = i + 1
                gamma_val = calculate_gamma_value(result['gamma'], X_train)
                result['gamma'] = f"{result['gamma']} ({gamma_val:.4f})"
                all_results.append(result)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            y_preds.extend(y_pred)
            y_tests.extend(y_test)
            y_probas.extend(y_proba)

            acc = accuracy_score(y_test, y_pred)
            all_fold_errors.append(1 - acc)
            all_conf_matrices.append(confusion_matrix(y_test, y_pred))

            if model.kernel == 'linear':
                imp = np.abs(model.coef_).mean(axis=0)
            else:
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                imp = result.importances_mean
            feature_importances.append(imp)

            progress = 20 + int((50 * i) / total_folds)
            progress_bar.progress(progress)

        # Evaluasi agregat
        avg_error = np.mean(all_fold_errors)
        avg_accuracy = 1 - avg_error
    
        y_tests = np.array(y_tests)
        y_preds = np.array(y_preds)
        y_probas = np.array(y_probas)

        status_text.text("🧾 Menyimpan hasil prediksi dan menghitung metrik evaluasi...")

        hasil_prediksi_df = pd.DataFrame({'Actual': y_tests, 'Predicted': y_preds})
        hasil_prediksi_df.to_excel('data_pred/data_prediksi_SVM.xlsx', index=False)

        precision = precision_score(y_tests, y_preds, average='binary' if binary_class else 'weighted')
        recall = recall_score(y_tests, y_preds, average='binary' if binary_class else 'weighted')
        f1 = f1_score(y_tests, y_preds, average='binary' if binary_class else 'weighted')
        report_df = pd.DataFrame(classification_report(y_tests, y_preds, target_names=le_ketereratan.classes_, output_dict=True)).transpose()
        conf_matrix = sum(all_conf_matrices)
        kappa = cohen_kappa_score(y_tests, y_preds)

        if binary_class:
            tn, fp, fn, tp = conf_matrix.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            roc_auc = roc_auc_score(y_tests, y_probas[:, 1])
        else:
            specificity = None
            npv = None
            roc_auc = roc_auc_score(y_tests, y_probas, multi_class='ovr')

        
        progress_bar.progress(80)
        status_text.text("📉 Membuat visualisasi evaluasi model...")
        mean_importance = np.mean(feature_importances, axis=0)
        feature_importance_df = pd.DataFrame({
            'Fitur': ['kategori_ipk', 'kategori_lama_studi', 'kerja tim', 'beasiswa'],
            'Importance': mean_importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Fitur'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance SVM (Avg of Folds)')
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        progress_bar.progress(85)
        plt.figure(figsize=(8, 6))
        if binary_class:
            fpr, tpr, _ = roc_curve(y_tests, y_probas[:, 1])
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        else:
            y_tests_binarized = label_binarize(y_tests, classes=np.unique(y))
            for i, color in zip(range(len(le_ketereratan.classes_)), ['blue', 'red', 'green', 'orange', 'purple']):
                fpr, tpr, _ = roc_curve(y_tests_binarized[:, i], y_probas[:, i])
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, label=f'Kelas {le_ketereratan.classes_[i]} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC-AUC SVM')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig("roc_curve.png")
        plt.close()

        progress_bar.progress(90)
        results_df = pd.DataFrame(all_results)
        kernel_best_detail_df = results_df.loc[results_df.groupby('kernel')['mean_test_accuracy'].idxmax()]
        kernel_best_detail_df = kernel_best_detail_df[['kernel', 'C', 'gamma', 'mean_test_accuracy']]
        kernel_best_detail_df.rename(columns={'mean_test_accuracy': 'best_accuracy'}, inplace=True)

        y_pred_labels = le_ketereratan.inverse_transform(y_preds)
        y_test_labels = le_ketereratan.inverse_transform(y_tests)

        progress_bar.progress(95)
        status_text.text("💾 Menyimpan model dan encoder...")

        result = {
            'accuracy': avg_accuracy,
            'classification_report': report_df,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'specificity': specificity,
            'npv': npv,
            'kappa': kappa,
            'feature_importance': feature_importance_df,
            'feature_importance_image': 'feature_importance.png',
            'roc_auc': roc_auc,
            'roc_image': 'roc_curve.png',
            'y_test_labels': y_test_labels,
            'y_pred_labels': y_pred_labels,
            'label_encoder': le_ketereratan,
            'average_error': avg_error,
            'average_accuracy': avg_accuracy,
            'jumlah_data_training': len(X_train),
            'jumlah_data_testing': len(X_test),
            'best_params': grid_search.best_params_,
            'grid_search_results': results_df,
            'best_accuracy_per_kernel': kernel_best_detail_df
        }

        save_model_to_db(
            model_object=model,
            kode_model=generate_kode_model(),
            kode_algoritma='SVM',
            jumlah_fold=n_splits,
            akurasi_model=avg_accuracy,
            id_pengguna=st.session_state.get("id_pengguna")
        )
        
        encoders_to_save = {
            'ENC001': ('le_ipk', le_ipk),
            'ENC002': ('le_studi', le_studi),
            'ENC003': ('le_beasiswa', le_beasiswa),
            'ENC004': ('le_kerjatim', le_kerjatim),
            'ENC005': ('le_ketereratan', le_ketereratan),
            'SLC001': ('scaler', scaler),
        }
        save_encoder_to_db(encoders_to_save)
        
        progress_bar.progress(100)
        status_text.text("✅ Proses selesai!")
        play_success_sound()
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        print("Model SVM telah disimpan.")

    return result

