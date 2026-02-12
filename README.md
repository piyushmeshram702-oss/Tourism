# Tourism Experience Analytics

A comprehensive analytics system for tourism data that includes:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Machine learning models for rating and visit mode prediction
- Recommendation system
- Interactive Streamlit web application

## Project Structure

```
tourism/
├── data/
│   ├── raw/              # Original Excel files
│   └── processed/        # Cleaned and processed data
├── src/
│   ├── data_loader.py    # Data loading and cleaning
│   ├── data_merger.py    # Data merging and feature engineering
│   ├── eda.py           # Exploratory Data Analysis
│   ├── models/
│   │   ├── regression.py
│   │   ├── classification.py
│   │   └── recommendation.py
│   └── visualization.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda_analysis.ipynb
│   └── 03_modeling.ipynb
├── app/
│   └── streamlit_app.py
├── main.py              # Main execution script
├── requirements.txt
└── README.md
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## Features

- **Data Processing**: Clean and merge multiple tourism datasets
- **EDA**: Comprehensive exploratory analysis with visualizations
- **Machine Learning**: 
  - Regression models for rating prediction
  - Classification models for visit mode prediction
- **Recommendation System**: Hybrid collaborative and content-based filtering
- **Web Interface**: Interactive dashboard for predictions and recommendations