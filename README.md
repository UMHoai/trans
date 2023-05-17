from sklearn.feature_extraction.text import CountVectorizer
import re

# Khởi tạo CountVectorizer và fit_transform với danh sách đoạn text
vectorizer = CountVectorizer(
    token_pattern=r'\b[a-zA-Z]+\b',
    stop_words=list(stop_words),
    analyzer='word',
    preprocessor=lambda text: re.sub(r'\d+', '', text)
)
X = vectorizer.fit_transform(text)

# Số lượng features
num_features = len(vectorizer.get_feature_names_out())
feature_name = vectorizer.get_feature_names_out()

print("Số lượng features có trong tập dữ liệu là:", num_features)
for feature in feature_name:
    print(feature)
