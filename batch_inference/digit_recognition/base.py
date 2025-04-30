import mlflow
import keras
from batch_inference.digit_recognition.cnn_utils import get_image_processor
from batch_inference.digit_recognition.data import transform_to_image

class DigitRecognizer(mlflow.pyfunc.PythonFunction):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def predict(self, context, model_input):
        if self.model is None:
            self.load_model()
        return self.model.predict(model_input)

    def train(self, x_train, y_train, **kwargs):
        """
        Train the model using the provided training data and parameters.

        :param x_train: The training features (input data).
        :param y_train: The training labels (target data).
        :param params: Optional parameters for training the model.
        """
        model = self.build_model()
        x_train = transform_to_image(x_train)
        optimizer = keras.optimizers.Adamax()
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        callbacks = [mlflow.keras.MlflowCallback()]

        with mlflow.start_run() as run:
            model.fit(
                x=x_train,
                y=y_train,
                batch_size=32,
                epochs=10,
                validation_split=0.2,
                callbacks=callbacks,
            )

    def _build_model(self):
        """
        Build the model using the get_image_processor function.

        This function creates a Keras model using the image processor defined in cnn_utils.
        The model is built using the functional API of Keras.
        """
        x_im_i, x_im = get_image_processor()
        model = keras.Model(inputs=x_im_i, outputs=x_im)
        return model

    def load_model(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, image):
        raise NotImplementedError("Subclasses should implement this method.")
