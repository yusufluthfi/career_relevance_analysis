                elif algoritma_select == "XGBoost":

                    if st.button("Jalankan XGBoost"):
                        # Catat waktu mulai
                        start_time = time.time()

                        results = run_xgboost(df_final)

                        # Catat waktu selesai
                        end_time = time.time()

                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        # Tampilkan durasi komputasi
                        st.write(f"**Waktu Komputasi XGBoost:** {elapsed_time:.2f} detik")
                        
                        st.write("### Hasil Evaluasi XGBoost")
                        
                        # Akurasi
                        st.write(f"**Akurasi Model:** {results['accuracy'] * 100:.2f}%")
                        
                        # Laporan Klasifikasi
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])

                        # Precision, Recall, dan F1-Score
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])

                        # ROC AUC Score
                        st.write("**ROC AUC Score:**")
                        st.write(results['roc_auc'])
                        
                        # Confusion Matrix
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        # Feature Importance
                        st.write("**Feature Importance:**")
                        st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)
                        
                        # Kurva ROC-AUC
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algoritma_select == "Naive Bayes":

                    if st.button("Jalankan Naive Bayes"):
                        # Catat waktu mulai
                        start_time = time.time()
                        results = run_naive_bayes(df_final, n_splits=jmlFold)
                        # Catat waktu selesai
                        end_time = time.time()
                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Naive Bayes</h3>",
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
                        st.write("**Laporan Klasifikasi:**")
                        col3, col4 = st.columns([0.6, 0.4])
                        with col3:
                            st.dataframe(results['classification_report'])
                        with col4:
                            st.write("**Precision:**", results['precision'])
                            st.write("**Recall:**", results['recall'])
                            st.write("**F1-Score:**", results['f1_score'])
                            st.write("**ROC AUC Score:**", results["roc_auc"])
                            st.write("**Fold Terbaik:**", results['fold_terbaik'])
                            st.write("**Data Training:**", results['jumlah_data_training'])
                            st.write("**Data Testing:**", results['jumlah_data_testing'])
                            

                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        # Feature Importance tidak tersedia untuk Naive Bayes, jadi bagian ini dikomentari

                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algoritma_select == "CatBoost":

                    if st.button("Jalankan Naive CatBoost"):
                        # Catat waktu mulai
                        start_time = time.time()

                        results = run_catboost(df_final)

                        # Catat waktu selesai
                        end_time = time.time()

                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Naive Bayes</h3>",
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
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])

                        # Precision, Recall, dan F1-Score
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])

                        # ROC AUC Score
                        st.write("**ROC AUC Score:**")
                        st.write(results['roc_auc'])
                        
                        # Confusion Matrix
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        # Feature Importance
                        st.write("**Feature Importance:**")
                        st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)
                        
                        # Kurva ROC-AUC
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algoritma_select == "Logistic Regression":

                    if st.button("Jalankan Logistic Regression"):
                        # Catat waktu mulai
                        start_time = time.time()

                        results = run_logistic_regression(df_final)

                        # Catat waktu selesai
                        end_time = time.time()

                        # Hitung durasi komputasi
                        elapsed_time = end_time - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Naive Bayes</h3>",
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
                        
                        # Laporan Klasifikasi
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])

                        # Precision, Recall, dan F1-Score
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])

                        # ROC AUC Score
                        st.write("**ROC AUC Score:**")
                        st.write(results['roc_auc'])
                        
                        # Confusion Matrix
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        # Kurva ROC-AUC
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)

                        # Feature Importance (aktifkan jika perlu)
                        # st.write("**Feature Importance:**")
                        # st.image(results['feature_importance_image'], caption="Feature Importance", use_container_width=True)
                elif algoritma_select == "AdaBoost":
                    if st.button("Jalankan AdaBoost"):
                        start_time = time.time()
                        results = run_adaboost(df_final)
                        end_time = time.time()

                        st.write(f"**Waktu Komputasi AdaBoost:** {end_time - start_time:.2f} detik")
                        st.write("### Hasil Evaluasi AdaBoost")
                        st.write(f"**Akurasi Model:** {results['accuracy'] * 100:.2f}%")
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])
                        st.write("**ROC AUC Score:**", results['roc_auc'])
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])
                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algoritma_select == "KNN":
                    if st.button("Jalankan KNN"):
                        start_time = time.time()
                        results = run_knn(df_final)
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        st.write(f"**Waktu Komputasi KNN:** {elapsed_time:.2f} detik")
                        st.write("### Hasil Evaluasi KNN")
                        st.write(f"**Akurasi Model:** {results['accuracy'] * 100:.2f}%")
                        st.dataframe(results['classification_report'])
                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])
                        st.write("**ROC AUC Score:**", results['roc_auc'])
                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algoritma_select == "Voting Classifier (KNN + CatBoost)":
                    if st.button("Jalankan Voting Classifier"):
                        # Catat waktu mulai
                        start_time = time.time()
                        results = run_voting_classifier(df_final)
                        # Hitung waktu komputasi
                        elapsed_time = time.time() - start_time

                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown(
                            "<h3 style='text-align: center;'>Hasil Evaluasi Voting Classifier</h3>",
                            unsafe_allow_html=True
                        )
                        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                        col1, col2 = st.columns([0.6, 0.4])
                        with col1:
                            st.write(f"***Akurasi Model:***  {results['accuracy'] * 100:.2f} %")
                        with col2:
                            st.write(f"***Waktu Komputasi:***  {elapsed_time:.2f} detik")

                        st.write("**Laporan Klasifikasi:**")
                        col3, col4 = st.columns([0.6, 0.4])
                        with col3:
                            st.dataframe(results['classification_report'])
                        with col4:
                            st.write("**Precision:**", results['precision'])
                            st.write("**Recall:**", results['recall'])
                            st.write("**F1-Score:**", results['f1_score'])
                            st.write("**ROC AUC Score:**", results["roc_auc"])

                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
                elif algoritma_select == "Voting AdaBoost + XGBoost":
                    if st.button("Jalankan Voting Classifier (AdaBoost + XGBoost)"):
                        start_time = time.time()

                        results = run_AdaBoost_XGBoost(df_final)

                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        st.write(f"**Waktu Komputasi Voting Classifier:** {elapsed_time:.2f} detik")
                        st.write("### Hasil Evaluasi Voting Classifier (AdaBoost + XGBoost)")

                        st.write(f"**Akurasi Model:** {results['accuracy'] * 100:.2f}%")
                        st.write("**Laporan Klasifikasi:**")
                        st.dataframe(results['classification_report'])

                        st.write("**Precision:**", results['precision'])
                        st.write("**Recall:**", results['recall'])
                        st.write("**F1-Score:**", results['f1_score'])
                        st.write("**ROC AUC Score:**", results['roc_auc'])

                        st.write("**Confusion Matrix:**")
                        st.write(results['confusion_matrix'])

                        st.write("**Kurva ROC-AUC:**")
                        st.image(results['roc_image'], caption="ROC Curve", use_container_width=True)
