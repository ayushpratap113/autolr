import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class AutoPreprocessor:
    def __init__(self, categorical_columns=None, numeric_columns=None, oversample=None):
        assert isinstance(categorical_columns, (list, type(None))), "categorical_columns must be a list or None"
        assert isinstance(numeric_columns, (list, type(None))), "numeric_columns must be a list or None"
        assert isinstance(oversample, (bool, type(None))), "oversample must be a boolean or None"

        # Initialize the instance variables
        self.__categorical_columns = categorical_columns
        self.__numeric_columns = numeric_columns
        self.__imputer = None
        self.__encoder = None
        self.__scaler = None
        self.__oversample = oversample

    def __fit(self, data, target_column):
        # If categorical or numeric columns are not provided, determine them based on the data type
        if self.__categorical_columns is None or self.__numeric_columns is None:
            self.__categorical_columns = []
            self.__numeric_columns = []

            for col in data.columns:
                if col != target_column:
                    if data[col].dtype == 'object':
                        self.__categorical_columns.append(col)
                    else:
                        self.__numeric_columns.append(col)

        # Fit the imputer on the data
        self.__imputer = SimpleImputer(strategy="most_frequent")
        self.__imputer.fit(data)

        # Fit the encoder on the categorical columns
        if self.__categorical_columns:
            self.__encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.__encoder.fit(data[self.__categorical_columns])

        # Fit the scaler on the numeric columns
        if self.__numeric_columns:
            self.__scaler = StandardScaler()
            self.__scaler.fit(data[self.__numeric_columns])

        # Determine if oversampling should be done based on the class distribution
        class_distribution = data[target_column].value_counts(normalize=True)
        if self.__oversample is None:
            if len(class_distribution) > 1 and class_distribution.min() / class_distribution.max() < 0.5:
                self.__oversample = True
            else:
                self.__oversample = False

    def __transform(self, data, target_column):
        # Apply the transformations to the data
        columns = data.columns
        data = self.__imputer.transform(data)
        data = pd.DataFrame(data, columns=columns)

        y = data[target_column]
        y = y.astype('int')

        # Apply the encoder to the categorical columns
        if self.__categorical_columns:
            encoded_data = self.__encoder.transform(data[self.__categorical_columns])
            encoded_data = pd.DataFrame(encoded_data, columns=self.__encoder.get_feature_names_out(self.__categorical_columns),
                                index=data.index)

        # Apply the scaler to the numeric columns
        if self.__numeric_columns:
            scaled_data = self.__scaler.transform(data[self.__numeric_columns])
            scaled_data = pd.DataFrame(scaled_data, columns=self.__numeric_columns, index=data.index)

        # Combine the transformed data
        if self.__categorical_columns and self.__numeric_columns:
            data = pd.concat([encoded_data, scaled_data], axis=1)
        elif self.__categorical_columns:
            data = encoded_data
        elif self.__numeric_columns:
            data = scaled_data

        data[target_column] = y

        return data

    def __fit_transform(self, data, target_column):
        # Fit the transformations and then apply them to the data
        self.__fit(data, target_column)
        return self.__transform(data, target_column)

    def __oversampling(self, x,y):
        # Apply oversampling if necessary
        if self.__oversample:
            ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = ros.fit_resample(x, y)
            return X_resampled, y_resampled
        return x,y

    def __split_data(self, data, target_column, test_size=0.2, random_state=42):
        # Split the data into training and testing sets
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, y_train = self.__oversampling(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def preprocess_data(self, data, target_column=None):
        # Check the types of the parameters
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if target_column is not None and not isinstance(target_column, str):
            raise TypeError("target_column must be a string or None")

        # preprocessing for data to be used in prediction
        if target_column is None:
            data['dummy_target'] = 0
            return self.__transform(data,'dummy_target')

        # Fit and transform the data, then split it into training and testing sets
        data = self.__fit_transform(data, target_column)
        return self.__split_data(data, target_column)