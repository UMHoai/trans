def generate_antonym(df, num_sentences=1, aug_min=1, aug_max=5, aug_p=0.3):
    # Define the AntonymAug replacement augmentation technique
    augmenter_antonym = naw.AntonymAug(aug_min=aug_min, aug_max=aug_max, aug_p = aug_p) 

    df_A = pd.DataFrame(columns=['sentence', 'class', 'method'])
    for _, row in df.iterrows():
        augmented_sentences = [row['sentence']] + [augmenter_antonym.augment(row['sentence']) for _ in range(num_sentences)]
        augmented_rows = pd.DataFrame({
            'sentence': augmented_sentences,
            'class': [row['class']] * len(augmented_sentences),
            'method': ['original'] + ['antonym'] * num_sentences
        })
        df_A = df_A.append(augmented_rows)
        df_A = df_A.drop_duplicates(subset='sentence')

    return df_A
