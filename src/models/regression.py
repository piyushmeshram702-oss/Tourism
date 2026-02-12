import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RatingPredictor:
    """Predict attraction ratings using regression models."""
    
    def __init__(self, data_path: str = "data/processed/merged_tourism_data.csv"):
        """
        Initialize rating predictor.
        
        Args:
            data_path (str): Path to merged data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
        self.load_data()
        self.prepare_features()
        
    def load_data(self):
        """Load the merged dataset."""
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data: {self.df.shape}")
        else:
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def prepare_features(self):
        """Prepare features for regression modeling with enhanced feature engineering."""
        if self.df is None:
            return
            
        # Select relevant features
        feature_columns = [
            'VisitYear', 'VisitMonth', 'ContinentId', 'RegionId', 
            'CountryId', 'CityId', 'AttractionTypeId', 'VisitMode'
        ]
        
        # Features that need encoding
        categorical_features = ['ContinentId', 'RegionId', 'CountryId', 'CityId', 'AttractionTypeId', 'VisitMode']
        
        # Create feature matrix
        X = self.df[feature_columns].copy()
        y = self.df['Rating'].copy()
        
        # Feature engineering: create additional meaningful features
        # Seasonal features
        X['Season'] = X['VisitMonth'].apply(lambda x: (x % 12 + 3) // 3)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
        
        # Period features
        X['IsPeakSeason'] = ((X['VisitMonth'].isin([6, 7, 8, 12])) * 1).astype(int)  # Summer/winter holidays
        
        # Interaction features
        X['CountryTypeInteraction'] = X['CountryId'] * X['AttractionTypeId']
        X['ModeTypeInteraction'] = X['VisitMode'] * X['AttractionTypeId']
        
        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(y.median())
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_features + ['Season']:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Test set shape: {self.X_test.shape}")
        
        self.label_encoders = label_encoders
        self.scaler = scaler
    
    def train_linear_regression(self):
        """Train Linear Regression model."""
        logger.info("Training Linear Regression model...")
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Evaluation metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        self.models['LinearRegression'] = model
        self.results['LinearRegression'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        logger.info(f"Linear Regression - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"Linear Regression - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        
        return model
    
    def train_random_forest(self):
        """Train Random Forest Regressor model with optimized parameters."""
        logger.info("Training Random Forest model with optimized parameters...")
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Evaluation metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        self.models['RandomForest'] = model
        self.results['RandomForest'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        logger.info(f"Random Forest - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"Random Forest - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        
        return model
    
    def train_xgboost(self):
        """Train XGBoost Regressor model with optimized hyperparameters."""
        logger.info("Training XGBoost regressor with optimized parameters...")
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        logger.info(f"XGBoost - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(f"XGBoost - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        
        return model
    
    def train_all_models(self):
        """Train all regression models."""
        self.train_linear_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        # Print comparison
        print("\n=== RATING PREDICTION MODEL COMPARISON ===")
        print(f"{'Model':<15} {'Train R²':<10} {'Test R²':<10} {'Train RMSE':<12} {'Test RMSE':<12}")
        print("-" * 60)
        
        for model_name, metrics in self.results.items():
            print(f"{model_name:<15} {metrics['train_r2']:<10.4f} {metrics['test_r2']:<10.4f} "
                  f"{metrics['train_rmse']:<12.4f} {metrics['test_rmse']:<12.4f}")
        
        print("=" * 60)
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
        print(f"\nBest model: {best_model} (Test R²: {self.results[best_model]['test_r2']:.4f})")
        
        return best_model
    
    def predict_rating(self, features_dict: dict, model_name: str = None) -> float:
        """
        Predict rating for given features.
        
        Args:
            features_dict (dict): Dictionary of feature values
            model_name (str): Name of model to use (default: best model)
            
        Returns:
            float: Predicted rating
        """
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_r2'])
            
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        # Prepare features
        feature_columns = [
            'VisitYear', 'VisitMonth', 'ContinentId', 'RegionId', 
            'CountryId', 'CityId', 'AttractionTypeId', 'VisitMode'
        ]
        
        # Create feature vector
        features = []
        for col in feature_columns:
            if col in features_dict:
                value = features_dict[col]
                # Apply same encoding as training data
                if col in self.label_encoders:
                    try:
                        value = self.label_encoders[col].transform([str(value)])[0]
                    except ValueError:
                        value = 0  # Unknown category
                features.append(value)
            else:
                features.append(0)  # Default value
        
        # Add engineered features
        visit_month = features_dict.get('VisitMonth', 1)
        country_id = features_dict.get('CountryId', 1)
        attraction_type_id = features_dict.get('AttractionTypeId', 1)
        visit_mode = features_dict.get('VisitMode', 1)
        
        # Seasonal features
        season = (visit_month % 12 + 3) // 3
        features.append(season)
        
        # Period features
        is_peak_season = 1 if visit_month in [6, 7, 8, 12] else 0
        features.append(is_peak_season)
        
        # Interaction features
        country_type_interaction = country_id * attraction_type_id
        features.append(country_type_interaction)
        
        mode_type_interaction = visit_mode * attraction_type_id
        features.append(mode_type_interaction)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(features_scaled)[0]
        
        # Clip prediction to valid range
        prediction = np.clip(prediction, 1, 5)
        
        return float(prediction)
    
    def get_feature_importance(self, model_name: str = 'RandomForest') -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name (str): Name of tree-based model
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name not in ['RandomForest', 'XGBoost']:
            raise ValueError("Feature importance only available for tree-based models")
            
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        feature_names = [
            'VisitYear', 'VisitMonth', 'ContinentId', 'RegionId', 
            'CountryId', 'CityId', 'AttractionTypeId', 'VisitMode'
        ]
        
        if model_name == 'RandomForest':
            importance = model.feature_importances_
        else:  # XGBoost
            importance = model.feature_importances_
            
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

# Example usage
if __name__ == "__main__":
    # Initialize and train models
    predictor = RatingPredictor()
    best_model = predictor.train_all_models()
    
    # Example prediction
    sample_features = {
        'VisitYear': 2023,
        'VisitMonth': 7,
        'ContinentId': 1,
        'RegionId': 1,
        'CountryId': 1,
        'CityId': 1,
        'AttractionTypeId': 13,  # Beaches
        'VisitMode': 3  # Family
    }
    
    predicted_rating = predictor.predict_rating(sample_features, best_model)
    print(f"\nPredicted rating: {predicted_rating:.2f}")
    
    # Show feature importance
    importance = predictor.get_feature_importance('RandomForest')
    print(f"\nFeature Importance:")
    print(importance.head(10))