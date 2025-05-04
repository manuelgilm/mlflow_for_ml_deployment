from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List


def get_sklearn_pipeline(
    numerical_features: List[str], categorical_features: List[str]
) -> Pipeline:
    """
    Get the sklearn pipeline for the diabetes prediction model.

    :param numerical_features: List of numerical feature names.
    :param categorical_features: List of categorical feature names.
    :return: A sklearn pipeline object.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier()),
        ]
    )
    return pipeline
