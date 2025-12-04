# **🧠 Brain Tumor Classification System (End-to-End MLOps)**

## **🚀 Live Demo**

Experience the full system in action through our interactive medical dashboard:

👉 [**NeuroScan AI Dashboard (Streamlit App)**](https://brain-tumor-classification-webapp.streamlit.app/)

The backend API is hosted on Hugging Face Spaces. You can explore the API documentation here:  
👉 API Swagger UI

## **📜 Project Overview**

This project is a comprehensive **End-to-End MLOps solution** for classifying brain tumors from MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

Going beyond simple model training, this project implements a complete production pipeline including:

1. **Advanced Deep Learning:** Comparing Custom CNNs vs. Transfer Learning strategies.  
2. **Production API:** A high-performance, asynchronous API built with FastAPI.  
3. **Containerization:** Fully dockerized application for consistent deployment.  
4. **CI/CD Pipeline:** Automated testing and deployment using GitHub Actions.  
5. **Observability:** Real-time error tracking and performance monitoring using **Sentry**.

## **🏗️ System Architecture (Microservices)**

The system follows a decoupled **Client-Server** architecture to ensure scalability and maintainability:

graph LR  
    User((User)) \--\>|Uploads MRI| Frontend\[Streamlit App\\n(Frontend Repo)\]  
    Frontend \--\>|POST /predict| Backend\[FastAPI Server\\n(Backend Repo)\]  
    Backend \--\>|Inference| Model\[EfficientNetB0\]  
    Backend \-.-\>|Logs & Errors| Sentry\[Sentry Monitoring\]  
    Backend \--\>|JSON Result| Frontend  
    Frontend \--\>|Visual Report| User

* **Frontend:** [GitHub Repo](https://github.com/abdoghazala7/brain-tumor-classification-webapp/tree/main) \- Hosted on Streamlit Cloud.  
* **Backend:** [GitHub Repo](https://github.com/abdoghazala7/brain-tumor-classification-inference-service) \- Hosted on Hugging Face Spaces (Docker).

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

## **📂 Project Structure**

├── .github/workflows/    \# CI/CD Pipeline configuration  
├── notebooks/            \# Jupyter notebooks for EDA & Training  
├── app.py                \# Streamlit Frontend code (in separate repo)  
├── config.py             \# Project configuration & Env vars  
├── main.py               \# FastAPI application entry point  
├── model\_loader.py       \# Logic for loading PyTorch model  
├── image\_utils.py        \# Image preprocessing logic  
├── predictor.py          \# Inference logic  
├── gunicorn\_config.py    \# Production server config  
├── Dockerfile            \# Docker container definition  
└── requirements.txt      \# Python dependencies

## **📄 License**

This project is licensed under the MIT License.

**Created with ❤️ by [Abdo Ghazala](https://www.google.com/search?q=https://github.com/abdoghazala7)**