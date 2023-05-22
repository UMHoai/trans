# CountVectorizer is used to convert text documents into numerical vectors
# Stop words have already been removed before this step

# Initialize the CountVectorizer with specified parameters
vectorizer = CountVectorizer(lowercase=True, stop_words=stop_words, max_features=max_features)

# Fit the vectorizer to the training data
vectorizer.fit(X_train)

# Transform the training data into vectors
review_vectors = vectorizer.transform(X_train)

# Convert the sparse matrix to a dense array
review_train = review_vectors.toarray()

# Print the length of the vocabulary (number of unique words)
print(len(vectorizer.vocabulary_))
