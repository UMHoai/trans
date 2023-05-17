vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Lấy danh sách các từ làm features
features = vectorizer.get_feature_names()

# In danh sách các từ
print("Danh sách các từ trong features:")
