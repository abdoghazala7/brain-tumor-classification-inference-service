import requests
import os

# API URL
API_URL = "http://127.0.0.1:8000/predict"

BASE_DIR = r"D:\gHaZaLa\Data Science Journy"
IMAGE_NAME = "test_image.jpg"
IMAGE_PATH = os.path.join(BASE_DIR, IMAGE_NAME)

def test_prediction(image_path):
    """
    Sends an image to the API and prints the prediction result.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at path: {image_path}")
        return

    try:
        with open(image_path, "rb") as image_file:
            files = {"file": ("test_image.jpg", image_file, "image/jpeg")}
            
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            print("\nPrediction Successful!")
            result = response.json()
            
            print("-" * 30)
            print(f"Filename: {result['filename']}")
            print(f"Prediction: {result['prediction']}")
            print("-" * 30)
            print("Confidence Scores:")
            for class_name, score in result['confidence_scores'].items():
                print(f"   - {class_name}: {score:.4f}")
            print("-" * 30)
            
        else:
            print(f"\nRequest Failed! Status Code: {response.status_code}")
            print(f"Message: {response.text}")

    except requests.exceptions.ConnectionError:
        print("\nConnection Error: Ensure the server (main.py) is running.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    test_prediction(IMAGE_PATH)