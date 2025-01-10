import uvicorn
from src.app import get_application

# Get the FastAPI application instance
app = get_application()

# Run the app with uvicorn, set to reload during development for auto-refresh
if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
