import os
from fastapi import APIRouter
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from fastapi.responses import JSONResponse

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
