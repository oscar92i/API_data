import os
from fastapi import APIRouter
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from fastapi.responses import JSONResponse
import pandas as pd

# Initialize the API router
router = APIRouter()

# Define the dataset download function
@router.get("/download-dataset")
async def download_dataset():
    try:
        # Set up paths
        dataset_path = "src/data"
        os.makedirs(dataset_path, exist_ok=True)
        
        # Initialize Kaggle API client
        api = KaggleApi()
        api.authenticate()

        # Download the Iris dataset
        dataset = "uciml/iris"
        api.dataset_download_files(dataset, path=dataset_path, unzip=True)

        # Return success message
        return JSONResponse(content={"message": "Dataset downloaded successfully!"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/load-dataset")
async def load_iris_dataset():
    try:
        # Set the path where the dataset is saved (from the download step)
        dataset_path = "src/data"

        # Construct the full path to the downloaded dataset
        file_path = os.path.join(dataset_path, "Iris.csv")  # The file might be named 'Iris.csv' or similar

        # Check if the file exists
        if not os.path.exists(file_path):
            return JSONResponse(content={"error": "Dataset file not found. Please download the dataset first."}, status_code=404)

        # Load the dataset using pandas
        df = pd.read_csv(file_path)

        # Convert the DataFrame to JSON and return it
        return JSONResponse(content=df.to_dict(orient="records"), status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)