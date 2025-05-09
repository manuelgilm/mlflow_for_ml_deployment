import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from typing import List
from utils.decorators import mlflow_tracking_uri
from utils.decorators import mlflow_client
from utils.decorators import mlflow_experiment


class CustomMLflowModel(mlflow.pyfunc.PythonModel):
    """ """

    def __init__(self, model_path: str):
        """
        Initialize the CustomMLflowModel with the path to the model.

        :param model_path: Path to the MLflow model.
        """
        self.model_path = model_path

    def load_context(self, context):
        """
        Load the model from the specified path.

        :param context: The context object containing the model path.
        """
        self.model = mlflow.pyfunc.load_model(self.model_path)

    def predict(self, context, model_input):
        """
        Perform prediction using the loaded model.

        :param context: The context object containing the model.
        :param model_input: Input data for prediction.
        :return: Predicted values.
        """
        return self.model.predict(model_input)


class WalmartSalesRegressor:
    """
    Custom MLflow model for sales regression.
    """

    def __init__(self):
        """
        Initialize the WalmartSalesRegressor.
        """
        self.numerical_features = ["Holiday_Flag"]
        self.categorical_features = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]

    @mlflow_tracking_uri
    @mlflow_experiment(name="walmart_sales_regression")
    @mlflow_client
    def fit_models(self, x_train, y_train):
        """
        Fits multiple models to the training data.

        :param x_train: Training features.
        :param y_train: Training target variable.
        """
        with mlflow.start_run(run_name="walmart-sales-regressors") as run:
            for store_id in x_train["Store"].unique():
                store_data = x_train[x_train["Store"] == store_id]
                store_target = y_train[y_train["Store"] == store_id]
                pipeline = self._get_sklearn_pipeline(
                    numerical_features=self.numerical_features,
                    categorical_features=self.categorical_features,
                )
                pipeline.fit(store_data, store_target)

                with mflow.start_run(
                    run_name=f"store-{store_id}", nested=True
                ) as store_run:
                    mlflow.sklearn.log_model(
                        pipeline,
                        artifact_path=f"model_store_{store_id}",
                        registered_model_name=f"sales-regressor-{store_id}",
                    )
                    mlflow.log_params({"store_id": store_id})

    def _get_sklearn_pipeline(
        self, numerical_features: List[str], categorical_features: List[str]
    ):
        """
        Get a scikit-learn pipeline for preprocessing and model training.

        :param numerical_features: List of numerical feature names.
        :param categorical_features: List of categorical feature names.
        :return: A scikit-learn pipeline object.
        """

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor()),
            ]
        )

        return pipeline

    def predict(self, store_id, x):
        """
        Predicts the target variable using the fitted model.
        """
        model_path = f"models:/walmart-sales-regressors/model_store_{store_id}/1"
        model = CustomMLflowModel(model_path=model_path)
        model.load_context(None)
        return model.predict(None, x)
