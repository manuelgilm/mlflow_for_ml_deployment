from usecases.digit_recognition.data import get_train_test_data
from usecases.digit_recognition.data import transform_to_image
import httpx
import pandas as pd
import json
import pickle
import base64
import numpy as np


def main() -> dict:
    """
    Perform online inference using a REST API.

    Args:
        image_path (str): Path to the input image.
        model_url (str): URL of the deployed model's REST API.

    Returns:
        dict: Inference results.
    """
    _, x_test, _, y_test = get_train_test_data()
    x_test = transform_to_image(x_test)

    url = "http://127.0.0.1:5000/invocations"
    n_samples = 1
    samples = x_test[0:n_samples]

    payload = {
        "instances": {"image_input": samples.tolist()},
    }
    headers = {"Content-Type": "application/json"}
    response = httpx.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        predictions = response.json().get("predictions")
        pred = np.argmax(predictions, axis=-1)
        print(pd.DataFrame({"predictions": pred, "y_test": y_test[0:n_samples]}))
        return
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")
