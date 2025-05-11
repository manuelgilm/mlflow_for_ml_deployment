from usecases.walmart_sales_regression.data import SalesDataProcessor
from usecases.walmart_sales_regression.base import WalmartSalesRegressor


def main():
    data_path = "../../Downloads/sales-walmart/Walmart_Sales.csv"
    data_processor = SalesDataProcessor(path=data_path)
    x_train, x_test, y_train, y_test = data_processor.create_train_test_split()
    print("Data loaded and split into training and testing sets.")

    # train model for three stores

    x_train = x_train[x_train["Store"].isin([1, 2, 3])]
    y_train = y_train[y_train["Store"].isin([1, 2, 3])]
    x_test = x_test[x_test["Store"].isin([1, 2, 3])]
    y_test = y_test[y_test["Store"].isin([1, 2, 3])]

    store_sales_regressor = WalmartSalesRegressor()
    store_sales_regressor.fit_models(x_train, y_train)
    print("Models fitted successfully.")
