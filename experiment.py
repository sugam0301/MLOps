import mlflow
logged_model = 'runs:/f942ca8f5d854ee5a10e6ab7fd704895/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))