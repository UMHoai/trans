https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/

https://medium.com/analytics-vidhya/an-introduction-to-multi-label-text-classification-b1bcb7c7364c

https://www.section.io/engineering-education/multi-label-classification-with-scikit-multilearn/

https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff

https://viblo.asia/p/multi-label-classification-cho-bai-toan-tag-predictions-oOVlY2Lr58W


def predict(text):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=maxlen)
    prediction = model.predict(padded_sequence)
    return prediction

# Example usage
text = "This is a toxic comment."
prediction = predict(text)
print(prediction)
