import os
from fastapi import APIRouter
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    
@router.get("/process-dataset")
async def process_iris_dataset():
    try:
        # Set the path where the dataset is saved (from the download step)
        dataset_path = "src/data"

        # Construct the full path to the downloaded dataset
        file_path = os.path.join(dataset_path, "Iris.csv")  # Ensure the dataset name is correct

        # Check if the file exists
        if not os.path.exists(file_path):
            return JSONResponse(content={"error": "Dataset file not found. Please download the dataset first."}, status_code=404)

        # Load the dataset using pandas
        df = pd.read_csv(file_path)

        # Check for any missing values (NaN) and handle them (e.g., drop them or fill them)
        if df.isnull().values.any():
            df = df.dropna()  # Dropping rows with missing values (you could also fill them with mean/median)

        # Encode the target variable (Species) if necessary
        df['Species'] = df['Species'].astype('category').cat.codes  # Using the correct column name 'Species'

        # Split the dataset into features (X) and target (y)
        X = df.drop("Species", axis=1)  # Features
        y = df["Species"]  # Target (encoded species)

        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize the features (optional, but often necessary for ML models)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Return the processed data (we can return summary info for now)
        return JSONResponse(content={
            "message": "Dataset processed successfully!",
            "train_samples": len(X_train_scaled),
            "test_samples": len(X_test_scaled),
            "features": list(X.columns)
        }, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)