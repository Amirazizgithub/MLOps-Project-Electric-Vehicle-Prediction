# MLOps-Project-Electric-Vehicle-Prediction

A comprehensive MLOps project designed to predict electric vehicle trends using machine learning techniques. This repository integrates data preprocessing, model training, evaluation, and deployment workflows, showcasing an end-to-end machine learning pipeline with DevOps practices. I register the trained ML models using MLflow and run the MLflow UI on AWS EC2 to track my models. Using a CI/CD pipeline, I deploy the project on AWS EC2.

---

## 1. Project Overview

This project aims to predict electric vehicle trends by leveraging machine learning models. It follows MLOps principles to ensure scalability, reproducibility, and maintainability. The pipeline includes:
- **Data Ingestion**: Loading and storing data into a MongoDB database.
- **Data Validation**: Ensuring data quality and consistency.
- **Data Transformation**: Preparing data for model training.
- **Model Training**: Building and training machine learning models.
- **Model Evaluation**: Assessing model performance.
- **Model Registering**: Using MLflow to register and track trained models.
- **Deployment**: Deploying the model for real-time predictions.

---

## 2. How to Use

### Prerequisites
- Python 3.10 or higher
- MongoDB installed and running
- Docker (optional, for containerization)
- AWS CLI configured with appropriate permissions
- GitHub repository with secrets configured for AWS deployment

### Steps to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/MLOps-Project-Electric-Vehicle-Prediction.git
   cd MLOps-Project-Electric-Vehicle-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script to store the dataset in MongoDB:
   ```bash
   python dataset_dumb_in_mongodb.py
   ```

4. Start the training pipeline:
   ```bash
   python app.py
   ```

5. Use Docker for containerized deployment:
   ```bash
   docker build -t electric-vehicle-prediction .
   docker run -p 5000:5000 electric-vehicle-prediction
   ```

---

## 3. Deploying on AWS ECR and EC2 Using GitHub Actions

### Prerequisites
- AWS account with ECR and EC2 services enabled.
- GitHub repository with the following secrets configured:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_REGION`
  - `AWS_ECR_REPO`
  - `EC2_INSTANCE_IP`
  - `EC2_USER`
  - `EC2_KEY`

### Steps for Deployment

1. **Build and Push Docker Image to AWS ECR**:
   - The GitHub Actions workflow builds the Docker image and pushes it to the specified AWS ECR repository.
   - Ensure the `AWS_ECR_REPO` secret in your GitHub repository contains the ECR repository URI.

2. **Deploy the Application on EC2**:
   - The workflow connects to the EC2 instance using SSH and pulls the Docker image from ECR.
   - The container is started on the EC2 instance.

3. **GitHub Actions Workflow**:
   - The CI/CD pipeline is defined in `.github/workflows/ci-cd.yaml`.
   - The workflow includes the following steps:
     - **Code Formatting Check**: Ensures the code adheres to the Black formatting standard.
     - **Unit Testing**: Runs tests defined in `tests/test_app_routes.py` using pytest.
     - **Docker Image Build and Push**: Builds the Docker image and pushes it to Amazon ECR.
     - **Deploy to EC2**: Connects to the EC2 instance and deploys the application.

4. **Triggering the Workflow**:
   - Push changes to the `main` branch or create a pull request to trigger the GitHub Actions workflow.

---

## 4. Project Structure

The project is organized as follows:

```
Electric_Vehicle_Prediction/
|-- Electric_Vehicle_Prediction/
|   |-- __init__.py                # Marks the directory as a Python package
|   |-- components/                # Core components of the pipeline
|   |   |-- __init__.py
|   |   |-- data_ingestion.py      # Handles data ingestion
|   |   |-- data_validation.py     # Validates the data
|   |   |-- data_transformation.py # Transforms data for training
|   |   |-- model_trainer.py       # Trains the machine learning model
|   |   |-- model_evaluation.py    # Evaluates the trained model
|   |-- configurations/            # Configuration files
|   |   |-- __init__.py
|   |   |-- mongo_db_connection.py # MongoDB connection setup
|   |-- constants/                 # Stores constant values
|   |   |-- __init__.py
|   |-- data_access/               # Data access layer
|   |   |-- __init__.py
|   |   |-- XYZ.py                 # Placeholder for data access logic
|   |-- entity/                    # Entity definitions
|   |   |-- __init__.py
|   |   |-- config_entity.py       # Configuration entities
|   |   |-- artifact_entity.py     # Artifact entities
|   |   |-- estimator.py           # Estimator logic
|   |-- exceptions/                # Custom exception handling
|   |   |-- __init__.py
|   |-- logger/                    # Logging utilities
|   |   |-- __init__.py
|   |-- pipeline/                  # Pipeline scripts
|   |   |-- __init__.py
|   |   |-- training_pipeline.py   # Training pipeline
|   |   |-- prediction_pipeline.py # Prediction pipeline
|   |-- utils/                     # Utility functions
|   |   |-- __init__.py
|   |   |-- main_utils.py          # Main utility functions
|-- mlflow/                        # MLflow tracking and artifacts
|   |-- models/model.py            # Model storage
|   |-- artifacts/                 # Artifacts directory
|   |-- tracking/mlflow_server.sh  # MLflow server script
|-- notebooks/                     # Jupyter notebooks for experimentation
|   |-- __init__.py
|   |-- mongo_db.ipynb             # MongoDB interaction notebook
|   |-- EDA_XYZ.ipynb              # Exploratory Data Analysis
|   |-- Feature_Engineering_and_Model_Training.ipynb # Feature engineering and training
|   |-- data_drift_demo_evidently.ipynb # Data drift demonstration
|-- static/                        # Static files for the web app
|   |-- CSS/
|       |-- style.css              # CSS styles
|-- templates/                     # HTML templates for the web app
|   |-- index.html                 # Main HTML template
|-- .github/                       # GitHub workflows for CI/CD
|   |-- workflows/
|       |-- ci-cd.yaml             # CI/CD pipeline configuration
|-- app.py                         # Main application script
|-- requirements.txt               # Python dependencies
|-- README.md                      # Project documentation
|-- .gitignore                     # Git ignore file
|-- .env                           # Environment variables
|-- .dockerignore                  # Docker ignore file
|-- Dockerfile                     # Docker configuration
|-- setup.py                       # Setup script for packaging
```

---

## 5. Key Features

- **End-to-End Pipeline**: Covers all stages from data ingestion to deployment.
- **MLOps Practices**: Implements CI/CD, version control, and containerization.
- **Scalable Architecture**: Modular design for easy scalability and maintenance.
- **Experimentation**: Includes Jupyter notebooks for exploratory data analysis and feature engineering.

---

## 6. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 7. Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Submit a pull request.

---

## 8. Contact

For any questions or issues, please contact:

**Author**: Amir Aziz  
**Email**: amirds0235@gmail.com