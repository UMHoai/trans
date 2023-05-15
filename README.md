# Thêm cột "class" với số thứ tự của câu
df['class'] = 'Class ' + (df.index + 1).astype(str)
