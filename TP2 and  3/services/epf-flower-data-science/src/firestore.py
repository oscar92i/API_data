import google.auth
from google.cloud import firestore
from google.auth.exceptions import DefaultCredentialsError


class FirestoreClient:
    """Wrapper around a Firestore database."""

    client: firestore.Client

    def __init__(self, service_account_json: str = None) -> None:
        """Initialize the Firestore client."""
        try:
            if service_account_json:
                # Use service account if provided
                self.client = firestore.Client.from_service_account_json(service_account_json)
            else:
                # Use default credentials
                credentials, _ = google.auth.load_credentials_from_file("src/config/api-data-49ac8-firebase-adminsdk-vjdi8-6e09bb3c0e.json")
                self.client = firestore.Client(credentials=credentials)
        except DefaultCredentialsError:
            raise Exception("Failed to authenticate. Ensure Google Cloud credentials are set.")

    def get(self, collection_name: str, document_id: str) -> dict:
        """Retrieve a document by its ID."""
        try:
            doc = self.client.collection(collection_name).document(document_id).get()
            if doc.exists:
                return doc.to_dict()
            raise FileNotFoundError(
                f"No document found in '{collection_name}' with ID '{document_id}'"
            )
        except Exception as e:
            raise Exception(f"Error getting document: {str(e)}")

    def add(self, collection_name: str, document_id: str, data: dict) -> dict:
        """Add a document to Firestore."""
        try:
            self.client.collection(collection_name).document(document_id).set(data)
            return {"message": "Document added successfully."}
        except Exception as e:
            raise Exception(f"Error adding document: {str(e)}")

    def update(self, collection_name: str, document_id: str, data: dict) -> dict:
        """Update an existing document."""
        try:
            self.client.collection(collection_name).document(document_id).update(data)
            return {"message": "Document updated successfully."}
        except Exception as e:
            raise Exception(f"Error updating document: {str(e)}")

    def delete(self, collection_name: str, document_id: str) -> dict:
        """Delete a document by its ID."""
        try:
            self.client.collection(collection_name).document(document_id).delete()
            return {"message": "Document deleted successfully."}
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")
