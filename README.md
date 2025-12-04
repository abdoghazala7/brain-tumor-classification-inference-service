# **🧠 Brain Tumor Classification System (End-to-End MLOps)**

## **🚀 Live Demo**

Experience the full system in action through our interactive medical dashboard:

👉 [**NeuroScan AI Dashboard (Streamlit App)**](https://brain-tumor-classification-webapp.streamlit.app/)

The backend API is hosted on Hugging Face Spaces. You can explore the API documentation here:
👉 **[API Swagger UI](https://abdoghazala7-brain-tumor-classification-api.hf.space/docs)**

## **📜 Project Overview**

This project is a comprehensive **End-to-End MLOps solution** for classifying brain tumors from MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

Going beyond simple model training, this project implements a complete production pipeline, including:

1. **Advanced Deep Learning:** Comparing Custom CNNs vs. Transfer Learning strategies.  
2. **Production API:** A high-performance, asynchronous API built with FastAPI.  
3. **Containerization:** Fully Dockerized application for consistent deployment.  
4. **CI/CD Pipeline:** Automated testing and deployment using GitHub Actions.  
5. **Observability:** Real-time error tracking and performance monitoring using **Sentry**.

 ## 🏗️ System Architecture (Microservices)

The system is designed using a robust **Client-Server** architecture, ensuring separation of concerns and scalability.
```mermaid
graph TD
    subgraph "Frontend (Streamlit)"
        User((👤 User)) -->|Uploads MRI Scan| UI[🖥️ Web Interface]
        UI -->|Sends Image (POST)| API_Call[📡 API Request]
    end

    subgraph "Backend (FastAPI & Docker)"
        API_Call -->|Receives Request| API[⚙️ FastAPI Server]
        API -->|Inference| Model[🧠 EfficientNetB0]
        Model -->|Prediction| API
    end

    subgraph "Monitoring & Logging"
        API -.->|Logs Metrics| Sentry[🚨 Sentry]
    end

    API -->|Returns JSON| UI
    UI -->|Displays Result| User

    style User fill:#f9f,stroke:#333,stroke-width:2px
    style UI fill:#bbf,stroke:#333,stroke-width:2px
    style API fill:#dfd,stroke:#333,stroke-width:2px
    style Model fill:#fdd,stroke:#333,stroke-width:2px
    style Sentry fill:#ddd,stroke:#333,stroke-width:2px
```


## **🧠 Model Development Journey**

We followed a rigorous scientific approach to achieve state-of-the-art results.

### **1\. Exploratory Data Analysis (EDA)**

* Analyzed class distribution to check for imbalance.  
* Visualized MRI samples to understand tumor features (texture, location).  
* Checked image dimensions to determine optimal resizing strategies.

### **2\. Approach A: Custom CNN (The Baseline)**

* **Architecture:** Built a deep Convolutional Neural Network from scratch.  
* **Techniques:** Used Conv2d, BatchNorm, MaxPooling, and Dropout (0.5) for regularization.  
* **Optimization:** Trained with AdamW optimizer and ReduceLROnPlateau scheduler.  
* **Result:** Achieved **\~95% Accuracy**.

### **3\. Approach B: Advanced Transfer Learning (The Winner) 🏆**

* **Base Model:** EfficientNetB0 (pre-trained on ImageNet) via timm.  
* **Strategy:** Implemented a **Three-Phase Gradual Fine-Tuning** strategy:  
  1. **Phase 1:** Freeze base model, train only the classifier head (lr=1e-2).  
  2. **Phase 2:** Unfreeze top convolutional blocks, fine-tune with lower rate (lr=1e-4).  
  3. **Phase 3:** Unfreeze entire model, ultra-low rate fine-tuning (lr=1e-5) for domain adaptation.

### **🥇 Final Results Comparison**

| Metric | Custom CNN | Transfer Learning (EfficientNet) |
| :---- | :---- | :---- |
| **Accuracy** | \~95% | **\~98%** |
| **AUC Score** | \~0.9950 | **\~0.9996** |
| **Model Size** | \~135 Million Params | **\~5.3 Million Params** |
| **Efficiency** | Heavy & Slow | **25x Smaller & Faster** |

**Conclusion:** The fine-tuned EfficientNetB0 not only achieved higher accuracy but is also significantly more efficient for production deployment.

## **⚙️ Backend Engineering (Production-Grade API)**

The backend is built to industry standards using **FastAPI**.

### **Key Engineering Features:**

* **Lifespan Management:** The AI model is loaded **once** into memory during startup (not per request), ensuring millisecond latency.  
* **Fail-Fast Mechanism:** If the model fails to load or files are missing, the server shuts down immediately (sys.exit(1)) to alert the orchestrator.  
* **Robust Validation:** Inputs are strictly validated for file type (MIME checking) and file size limits (Max 5MB) to prevent crashes.  
* **Concurrency:** Uses **Gunicorn** as a process manager with dynamic worker calculation based on CPU cores.  
* **Logging:** Structured logging system (instead of print) for easier debugging.

## **🛠️ Infrastructure & MLOps**

### **1\. Dockerization 🐳**

* **Multi-Stage Build:** Used a 2-stage Dockerfile to keep the final image lightweight by discarding build dependencies.  
* **Security:** The container runs as a **non-root user** (appuser) to prevent security breaches.  
* **Optimization:** Used python:3.10-slim and installed CPU-only versions of PyTorch to reduce image size.  
* **Health Checks:** Implemented a HEALTHCHECK instruction to restart the container if the API becomes unresponsive.

### **2\. CI/CD Pipeline (GitHub Actions) 🤖**

Automated workflow triggers on every push to the main branch:

1. **Build:** Builds the Docker image.  
2. **Test:** Runs a containerized Integration Test using a python client script (test\_api.py) to verify API endpoints.  
3. **Deploy:** If tests pass, it pushes the code to **Hugging Face Spaces** using Git LFS for large model files.  
4. **Verify:** Polls the live server status until it returns RUNNING.

### **3\. Monitoring with Sentry 🚨**

* Integrated **Sentry SDK** to track runtime errors and performance issues in production.  
* Configured to capture user impact and stack traces instantly.

## **💻 How to Run Locally**

### **Prerequisites**

* Python 3.10+  
* Docker (Optional but recommended)

### **Method 1: Using Python (Direct)**

1. **Clone the repository:**  
   git clone \[https://github.com/abdoghazala7/brain-tumor-classification-inference-service.git\](https://github.com/abdoghazala7/brain-tumor-classification-inference-service.git)  
   cd brain-tumor-classification-inference-service

2. **Install dependencies:**  
   pip install \-r requirements.txt

3. Set up Environment:  
   Create a .env file:  
   MODEL\_NAME=tf\_efficientnet\_lite0  
   MODEL\_PATH=efficientnet\_finetuned\_final.pth  
   \# Optional: SENTRY\_DSN=your\_dsn\_here

4. **Run the Server:**  
   python main.py

5. **Access API:** Go to http://localhost:8000/docs

### **Method 2: Using Docker (Production Simulation)**

\# Build the image  
docker build \-t brain-api .

\# Run the container (Mapping port 8000 host \-\> 7860 container)  
docker run \-d \-p 8000:7860 \--name brain-app brain-api

## 📂 Project Structure

The project is organized into a decoupled architecture with separate repositories for the backend API and the frontend application.

### **Backend Repository (Inference Service)**
This repository handles the model serving, API logic, and dockerization.

```text
brain-tumor-classification-inference-service/
│
├── .github/workflows/
│   └── ci_cd_pipeline.yml       # GitHub Actions workflow for automated testing & deployment
│
├── notebooks/
│   └── Brain Tumor MRI Classification.ipynb  # Research & Training notebook (EDA, Training, Eval)
│
├── .dockerignore                # Files to exclude from Docker build context
├── .gitignore                   # Files to exclude from Git tracking
├── Dockerfile                   # Instructions to build the production Docker image
├── README.md                    # Project documentation
├── config.py                    # Configuration settings & environment variables loader
├── efficientnet_finetuned_final.pth  # The trained PyTorch model weights
├── gunicorn_config.py           # Gunicorn server configuration (workers, timeout)
├── image_utils.py               # Utility functions for image preprocessing
├── main.py                      # Main FastAPI application entry point
├── model_loader.py              # Logic for loading and initializing the model
├── predictor.py                 # Core inference logic (prediction function)
├── requirements.txt             # Python dependencies for the backend
└── test_api.py                  # Client script for testing the API locally
```

## **📄 License**

This project is licensed under the MIT License.
