# Đoạn code để phân cụm và gán nhãn
kmeans2 = KMeans(n_clusters=5, random_state=13)
kmeans2.fit_predict(review_umapped)
kmeans2.fit(review_vectors)
cluster_labels = kmeans2.labels_

# Tạo DataFrame mới chứa dữ liệu đã phân cụm và gán nhãn
clustered_data = pd.DataFrame({
    'review': X_train,
    'sentiment': y_train,
    'cluster_label': cluster_labels
})

# Lưu DataFrame vào file CSV
clustered_data.to_csv('clustered_data.csv', index=False)
