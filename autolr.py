import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from preprocessing import AutoPreprocessor


class AutoLogisticRegression:
    def __init__(self, data_path:str, target_column:str, num_folds=5):
        assert isinstance(data_path, str), "data_path must be a string"
        assert isinstance(target_column, str), "target_column must be a string"
        assert isinstance(num_folds, int), "num_folds must be an integer"

        self.__data_path = data_path
        self.__target_column = target_column
        self.__num_folds = num_folds
        self.__best_model = None
        self.__data = pd.read_csv(self.__data_path)
        self.__column_len = len(self.__data.columns)


    def __logistic_regression(self, X_train, y_train):
        param_grid = {'Cs': [1, 10, 100], 'solver': 'liblinear'}

        logistic_model = LogisticRegressionCV(Cs=param_grid['Cs'], cv=self.__num_folds, random_state=42, n_jobs=-1, solver=param_grid['solver'])   
        logistic_model.fit(X_train, y_train)

        self.__best_params = logistic_model.C_
        self.__best_model = logistic_model
    


    def __evaluate_best_model(self, X_test, y_test):
        if self.__best_model is None:
            raise Exception("You must train the model before making predictions!")

        y_pred = self.__best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f'Accuracy: {accuracy}')
        print(f'Classification Report:\n{report}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        print(self.__feature_importance())
    
    def __feature_importance(self):
        if self.__best_model is None:
            raise Exception("You must train the model before getting feature importance!")

        # Get feature names
        feature_names = self.__feature_names

        # Get feature importance
        importance = self.__best_model.coef_[0]

        # Combine feature names and importance into a DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })

        # Sort by absolute value of importance
        feature_importance = feature_importance.reindex(feature_importance.importance.abs().sort_values(ascending=False).index)

        return feature_importance

    def train(self):
        if self.__target_column not in self.__data.columns:
            raise ValueError(f"target_column {self.__target_column} does not exist in the DataFrame")

        self.__preprocessor = AutoPreprocessor()
        X_train, X_test, y_train, y_test = self.__preprocessor.preprocess_data(self.__data, self.__target_column)
        # self.find_best_model(X_train, y_train)
        self.__feature_names = X_train.columns.to_list()
        self.__logistic_regression(X_train, y_train)
        self.__evaluate_best_model(X_test, y_test)


    def predict(self, data_path:str):
        assert isinstance(data_path, str), "data_path must be a string"

        if self.__best_model is None:
            raise Exception("You must train the model before making predictions!")

        data = pd.read_csv(data_path)
        

        if self.__column_len - 1 != len(data.columns):
            raise Exception("Columns of test data does not match train data!")
        
        pred = self.__preprocessor.preprocess_data(data)
        pred = pred.drop('dummy_target', axis=1)
        # self.find_best_model(X_train, y_train)
        y_pred = self.__best_model.predict(pred)
        return y_pred
