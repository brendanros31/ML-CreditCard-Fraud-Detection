from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Standard scaler
def scale_features(df, df_col1, df_col2):
    scaler = StandardScaler()
    df[[df_col1, df_col2]] = scaler.fit_transform(df[[df_col1, df_col2]])
    return df, scaler


# train test split
def split(X, y, test_size=0.3, random_state=42, show=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if show==True:
        print(f'Shape train_X: {X_train}')
        print(f'Shape train_Y: {y_train}')
        
    return X_train, X_test, y_train, y_test
