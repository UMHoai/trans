from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from spacy.lang.de.stop_words import STOP_WORDS

# Tạo pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer(lowercase=True, stop_words=list(STOP_WORDS))),
    ('clf', LogisticRegression())
])

# Thiết lập các giá trị max_features cần thử nghiệm
param_grid = {
    'vect__max_features': [1000, 2000, 3000, 4000, 5000]
}

# Thiết lập GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1)

# Huấn luyện GridSearchCV trên dữ liệu
grid_search.fit(X_train, y_train)

# In ra giá trị tối ưu của max_features
print("Best max_features:", grid_search.best_params_['vect__max_features'])
