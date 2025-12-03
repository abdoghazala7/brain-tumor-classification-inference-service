import requests
import sys
import io
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

def create_dummy_image():
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_prediction():
    print(f"Starting Test on: {API_URL}")

    try:
        image_file = create_dummy_image()
        
        files = {"file": ("test_image.jpg", image_file, "image/jpeg")}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            
            if "prediction" in result and "confidence_scores" in result:
                print("\nTest Passed!")
                print(f"Filename: {result['filename']}")
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence_scores']}")
                sys.exit(0)
            else:
                print("\nTest Failed: Invalid JSON response structure.")
                sys.exit(1)
            
        else:
            print(f"\nTest Failed! Status Code: {response.status_code}")
            print(f"Message: {response.text}")
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        print("\nConnection Error: Could not connect to server.")
        print("Make sure the FastAPI server is running locally on port 8000.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_prediction()