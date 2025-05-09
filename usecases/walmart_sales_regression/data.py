from pathlib import Path
from typing import Optional
from typing import Union
import pandas as pd


class SalesDataProcessor:
    def __init__(self, path):
        """
        Initialize the SalesDataProcessor with the path to the data and the data itself.

        :param path: Path to the data file.
        """
        self.load_data(path)

    def load_data(self, path: Union[str, Path]) -> None:
        """
        Load the data from the specified path.

        :return: DataFrame containing the loaded data.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        self.df = pd.read_csv(path)
        self.data = self.df.copy()

    def create_train_test_data(
        self, store_id: Optional[int] = None, test_size: Optional[float] = 0.2
    ) -> None:
        """
        Create training and testing data from the loaded dataset.
        If store_id is provided, filter the data for that store.
        """
        sales_data = self.data.copy()
        sales_data["Date"] = pd.to_datetime(sales_data["Date"])
        sales_data.groupby("Store")["Weekly_Sales"].sum().reset_index()
        self.store_sales = {}
        self.store_sales[store_id] = (
            self.data[self.data["Store"] == store_id][["Weekly_Sales", "Date"]]
            .groupby("Date")
            .sum()
        )

    def preprocess(self):
        # Example preprocessing steps
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data["Sales"] = self.data["Sales"].astype(float)
        return self.data
