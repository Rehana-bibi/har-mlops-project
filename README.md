# har-mlops-project
# Human Activity Recognition MLOps Project

This project implements an MLOps pipeline for Human Activity Recognition using smartphone sensor data.

## Project Structure
- `.github/workflows`: CI/CD pipeline configurations
- `data`: Raw and processed data storage
- `src`: Source code for data processing, model training, and evaluation
- `tests`: Unit tests
- `config`: Configuration files

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure Azure credentials in config/model_config.yaml
4. Run tests: `pytest tests/`

## Dataset Setup & Usage

1. Dataset Preparation:
   - Your `data` folder should have the following structure:
     ```
     data/
     ├── raw/
     │   ├── test/
     │   └── train/
     ├── processed/
     └── data_versioning/
     ```

2. Dataset Download and Setup:
   - Download the UCI HAR Dataset from: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
   - Extract the downloaded zip file
   - Copy the `test` and `train` folders into the `data/raw` directory of this project

3. Running the Pipeline:
   - Process the raw data:
     ```bash
     python src/data/make_dataset.py
     ```
   - Generate features:
     ```bash
     python src/features/build_features.py
     ```
   - Train the model:
     ```bash
     python src/models/train_model.py
     ```
   - Evaluate results:
     ```bash
     python src/models/evaluate_model.py
     ```

Note: The raw dataset files are not included in the repository due to size constraints. Please download them separately following the instructions above.

## Pipeline Steps
1. Data Processing
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Performance Monitoring
