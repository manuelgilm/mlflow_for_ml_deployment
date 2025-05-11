import mlflow
from mlflow.types import Schema
from mlflow.types.schema import ColSpec
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ParamSpec
from mlflow.types.schema import ParamSchema
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


class WalmartSalesRegressor(mlflow.pyfunc.PythonModel):
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
    def fit_models(self, x_train, y_train, **kwargs):
        """
        Fits multiple models to the training data.

        :param x_train: Training features.
        :param y_train: Training target variable.
        """
        with mlflow.start_run(run_name="walmart-sales-regressors") as run:

            for store_id in x_train["Store"].unique():
                self.fit_model(
                    x_train=x_train,
                    y_train=y_train,
                    store_id=store_id,
                    parent_run_id=run.info.run_id,
                )

            # get the model signature
            signature = self._get_model_signature()
            # log the entire class as a model
            mlflow.pyfunc.log_model(
                artifact_path="walmart-store-sales-regressor",
                python_model=self,
                registered_model_name="walmart-store-sales-regressor",
                signature=signature,
            )

            # set the model version alias to "production"
            model_version = mlflow.search_model_versions(
                filter_string="name='walmart-store-sales-regressor'",
                max_results=1,
            )[0]
            client = kwargs["mlflow_client"]
            client.set_registered_model_alias(
                name="walmart-store-sales-regressor",
                version=model_version.version,
                alias="production",
            )

    def fit_model(self, x_train, y_train, store_id: int, parent_run_id: str):
        """
        Fits a single model to the training data for a specific store.

        :param x_train: Training features.
        :param y_train: Training target variable.
        :param store_id: The store ID for which to fit the model.
        """
        store_data = x_train[x_train["Store"] == store_id]
        store_target = y_train[y_train["Store"] == store_id]

        pipeline = self._get_sklearn_pipeline(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
        )

        pipeline.fit(store_data, store_target)

        with mlflow.start_run(
            run_name=f"run_store_{store_id}", parent_run_id=parent_run_id, nested=True
        ):
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=f"model_store_{store_id}",
            )
            mlflow.log_params({"store_id": store_id})

    def _get_model_signature(self) -> ModelSignature:
        """
        Get the model signature for the MLflow model.

        :return: Model signature object.
        """
        feature_specification = [
            ColSpec(type="integer", name="Holiday_Flag"),
            ColSpec(type="float", name="Temperature"),
            ColSpec(type="float", name="Fuel_Price"),
            ColSpec(type="float", name="CPI"),
            ColSpec(type="float", name="Unemployment"),
        ]

        param_specification = [
            ParamSpec(dtype="integer", name="store_id", default=1),
        ]
        param_schema = ParamSchema(
            params=param_specification,
        )
        input_schema = Schema(inputs=feature_specification)
        output_schema = Schema(
            inputs=[ColSpec(type="float", name="Weekly_Sales")],
        )
        signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
            params=param_schema,
        )

        return signature

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

    def predict(self, store_id: int, stage: str, x):
        """
        Predicts the target variable using the fitted model.

        :param store_id: The store ID for which to make predictions.
        :param stage: The stage of the model (e.g., "production").
        :param x: The input data for prediction.
        :return: The predicted values.
        """
        model_path = f"models:/sales-regressor-{store_id}@{stage}"
        model = mlflow.sklearn.load_model(model_path)
        predictions = model.predict(x)
        return predictions
