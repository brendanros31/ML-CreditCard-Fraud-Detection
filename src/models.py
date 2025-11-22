from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Build models
def build_model(model_type, params=None):
    if model_type == 'DecisionTree':
        return DecisionTreeClassifier(**(params or {}))
    
    elif model_type == 'RandomForest':
        return RandomForestClassifier(**(params or {}))
    
    elif model_type =='XGBoost':
        return XGBClassifier(**(params or {}))


# Train model
def train_model(model, X_train, y_train, ):
    return model.fit(X_train, y_train)