from sklearn.feature_extraction.text import CountVectorizer

# Dữ liệu huấn luyện
X_train = ['Tôi thích học máy', 'Máy tính là một công cụ hữu ích', 'Học máy là thú vị']

# Tạo danh sách các giá trị max_features để thử nghiệm
max_features_list = [10, 20, 100, 200, 300]

best_max_features = None
best_vocabulary_size = 0

# Thử nghiệm các giá trị max_features và lưu giá trị tốt nhất
for max_features in max_features_list:
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_vectors = vectorizer.fit_transform(X_train)

    vocabulary_size = len(vectorizer.vocabulary_)
    if vocabulary_size > best_vocabulary_size:
        best_vocabulary_size = vocabulary_size
        best_max_features = max_features

print("Best max_features:", best_max_features)
