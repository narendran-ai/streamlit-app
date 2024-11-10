from sklearn.preprocessing import LabelEncoder

def encode(df):
    encoder = LabelEncoder()
    for columns in range(len(df.columns)):
        df[df.columns[columns]]=encoder.fit_transform(df[df.columns[columns]])
    return df