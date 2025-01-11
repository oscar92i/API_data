import os
from fastapi import APIRouter
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastapi import HTTPException
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

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
    
@router.get("/split-dataset")
async def split_iris_dataset():
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

        # Convert the numpy arrays back to lists for easier JSON serialization
        X_train_list = X_train_scaled.tolist()
        X_test_list = X_test_scaled.tolist()
        y_train_list = y_train.tolist()
        y_test_list = y_test.tolist()

        # Return the processed and split data as a JSON response
        return JSONResponse(content={
            "message": "Dataset split into train and test successfully!",
            "train_data": {
                "features": X_train_list,
                "labels": y_train_list
            },
            "test_data": {
                "features": X_test_list,
                "labels": y_test_list
            }
        }, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# Function to load model parameters
def load_model_params():
    config_path = os.path.join("src\config\model_parameters.json")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config['LogisticRegression']
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model parameters config file not found.")
    
@router.post("/train-model")
async def train_model():
    try:
        # Define paths
        data_path = "src/data/iris.csv"
        model_path = "src/models/iris_model.joblib"
        
        # Load dataset
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Dataset not found. Please download and process it first.")
        df = pd.read_csv(data_path)
        
        if 'Species' not in df.columns:
            raise HTTPException(status_code=400, detail=f"'Species' column not found. Available columns are: {list(df.columns)}")
        
        # Prepare data
        X = df.drop(columns=['Species'])
        y = df['Species']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            "message": "Model trained and saved successfully!",
            "model_path": model_path,
            "accuracy": accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))