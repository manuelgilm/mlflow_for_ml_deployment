from utils.decorators import mlflow_tracking_uri
from utils.decorators import mlflow_client
from utils.decorators import mlflow_experiment
from batch_inference.digit_recognition.data import get_train_test_data
from batch_inference.digit_recognition.cnn_utils import get_image_processor
from batch_inference.digit_recognition.data import transform_to_image
import keras
import mlflow


@mlflow_tracking_uri
@mlflow_experiment(
    name="digit_recognition",
    tags={"topic": "batch_inference", "level": "basic"},
)
@mlflow_client
def main(**kwargs) -> None:
    """ """

    x_train, x_test, y_train, y_test = get_train_test_data()
    x_train = transform_to_image(x_train)
    x_test = transform_to_image(x_test)

    # building the model
    input_name = "image_input"
    x_im_i, x_im = get_image_processor(input_name=input_name)
    model = keras.Model(inputs=x_im_i, outputs=x_im)

    optimizer = keras.optimizers.Adamax()
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [mlflow.keras.MlflowCallback()]

    with mlflow.start_run() as run:
        model.fit(
            x={input_name: x_train},
            y=y_train,
            validation_data=({input_name: x_test}, y_test),
            batch_size=32,
            epochs=10,
            validation_split=0.2,
            callbacks=callbacks,
        )

        # log model
        registered_model_name = "Digit_Recognition_Model"
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        # set model version alias to "production"
        model_version = mlflow.search_model_versions(
            filter_string=f"name='{registered_model_name}'", max_results=1
        )[0]
        client = kwargs["mlflow_client"]
        client.set_registered_model_alias(
            name=registered_model_name,
            version=model_version.version,
            alias="production",
        )
