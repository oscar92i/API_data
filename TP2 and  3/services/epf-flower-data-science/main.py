import uvicorn
from src.app import get_application
from src.api.routes import data


# Get the FastAPI application instance
app = get_application()
app.include_router(data.router)
# Run the app with uvicorn, set to reload during development for auto-refresh
if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8080)
