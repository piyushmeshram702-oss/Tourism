import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisitModePredictor:
    """Predict visit mode using classification models."""
    
    def __init__(self, data_path: str = "data/processed/merged_tourism_data.csv"):
        """
        Initialize visit mode predictor.
        
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
        """Prepare features for classification modeling with enhanced feature engineering."""
        if self.df is None:
            return
            
        # Select relevant features for visit mode prediction
        feature_columns = [
            'VisitYear', 'VisitMonth', 'Rating', 'ContinentId', 'RegionId', 
            'CountryId', 'CityId', 'AttractionTypeId', 'AttractionCountryId'
        ]
        
        # Features that need encoding
        categorical_features = ['ContinentId', 'RegionId', 'CountryId', 'CityId', 'AttractionTypeId', 'AttractionCountryId']
        
        # Create feature matrix
        X = self.df[feature_columns].copy()
        y = self.df['VisitMode'].copy()
        
        # Feature engineering: create additional meaningful features
        # Seasonal features
        X['Season'] = X['VisitMonth'].apply(lambda x: (x % 12 + 3) // 3)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
        
        # Period features
        X['IsPeakSeason'] = ((X['VisitMonth'].isin([6, 7, 8, 12])) * 1).astype(int)  # Summer/winter holidays
        
        # Interaction features
        X['CountryTypeInteraction'] = X['CountryId'] * X['AttractionTypeId']
        X['RatingTypeInteraction'] = X['Rating'] * X['AttractionTypeId']
        
        # Adjust labels to start from 0 for sklearn compatibility
        y = y - 1  # Convert from [1,2,3,4,5] to [0,1,2,3,4]
        
        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(y.mode()[0] if not y.mode().empty else 1)
        
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
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store original labels mapping
        self.label_mapping = {i: i+1 for i in range(5)}  # 0->1, 1->2, 2->3, 3->4, 4->5
        self.reverse_label_mapping = {i+1: i for i in range(5)}  # 1->0, 2->1, 3->2, 4->3, 5->4
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Test set shape: {self.X_test.shape}")
        logger.info(f"Class distribution in training set: {self.y_train.value_counts().to_dict()}")
        
        self.label_encoders = label_encoders
        self.scaler = scaler
    
    def train_random_forest(self):
        """Train Random Forest Classifier model with optimized parameters."""
        logger.info("Training Random Forest classifier with optimized parameters...")
        
        model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Evaluation metrics
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        train_precision = precision_score(self.y_train, y_pred_train, average='weighted')
        test_precision = precision_score(self.y_test, y_pred_test, average='weighted')
        train_recall = recall_score(self.y_train, y_pred_train, average='weighted')
        test_recall = recall_score(self.y_test, y_pred_test, average='weighted')
        train_f1 = f1_score(self.y_train, y_pred_train, average='weighted')
        test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
        
        self.models['RandomForest'] = model
        self.results['RandomForest'] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'test_f1': test_f1
        }
        
        logger.info(f"Random Forest - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Random Forest - Test F1 Score: {test_f1:.4f}")
        
        return model
    
    def train_xgboost(self):
        """Train XGBoost Classifier model with optimized parameters."""
        logger.info("Training XGBoost classifier with optimized parameters...")
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            objective='multi:softprob'
        )
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Evaluation metrics
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        train_precision = precision_score(self.y_train, y_pred_train, average='weighted')
        test_precision = precision_score(self.y_test, y_pred_test, average='weighted')
        train_recall = recall_score(self.y_train, y_pred_train, average='weighted')
        test_recall = recall_score(self.y_test, y_pred_test, average='weighted')
        train_f1 = f1_score(self.y_train, y_pred_train, average='weighted')
        test_f1 = f1_score(self.y_test, y_pred_test, average='weighted')
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'test_f1': test_f1
        }
        
        logger.info(f"XGBoost - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"XGBoost - Test F1 Score: {test_f1:.4f}")
        
        return model
    
    def train_all_models(self):
        """Train all classification models."""
        self.train_random_forest()
        self.train_xgboost()
        
        # Print comparison
        print("\n=== VISIT MODE PREDICTION MODEL COMPARISON ===")
        print(f"{'Model':<15} {'Train Acc':<10} {'Test Acc':<10} {'Train F1':<10} {'Test F1':<10}")
        print("-" * 55)
        
        for model_name, metrics in self.results.items():
            print(f"{model_name:<15} {metrics['train_accuracy']:<10.4f} {metrics['test_accuracy']:<10.4f} "
                  f"{metrics['train_f1']:<10.4f} {metrics['test_f1']:<10.4f}")
        
        print("=" * 55)
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['test_f1'])
        print(f"\nBest model: {best_model} (Test F1: {self.results[best_model]['test_f1']:.4f})")
        
        # Show confusion matrix for best model
        self.show_confusion_matrix(best_model)
        
        return best_model
    
    def show_confusion_matrix(self, model_name: str = None):
        """
        Show confusion matrix for a model.
        
        Args:
            model_name (str): Name of model to evaluate
        """
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_f1'])
            
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        # Create confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        print(f"\nConfusion Matrix for {model_name}:")
        print("Rows: True labels, Columns: Predicted labels")
        print(cm)
        
        # Show class mapping
        unique_modes = sorted(self.y_test.unique())
        mode_names = ['Business', 'Couples', 'Family', 'Friends', 'Solo']
        print(f"\nVisit Mode Mapping:")
        for i, mode_id in enumerate(unique_modes):
            mode_name = mode_names[mode_id-1] if mode_id <= len(mode_names) else f"Mode {mode_id}"
            print(f"  {mode_id}: {mode_name}")
    
    def predict_visit_mode(self, features_dict: dict, model_name: str = None) -> int:
        """
        Predict visit mode for given features.
        
        Args:
            features_dict (dict): Dictionary of feature values
            model_name (str): Name of model to use (default: best model)
            
        Returns:
            int: Predicted visit mode ID
        """
        if model_name is None:
            model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_f1'])
            
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        # Prepare features
        feature_columns = [
            'VisitYear', 'VisitMonth', 'Rating', 'ContinentId', 'RegionId', 
            'CountryId', 'CityId', 'AttractionTypeId', 'AttractionCountryId'
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
        rating = features_dict.get('Rating', 3.0)
        
        # Seasonal features
        season = (visit_month % 12 + 3) // 3
        features.append(season)
        
        # Period features
        is_peak_season = 1 if visit_month in [6, 7, 8, 12] else 0
        features.append(is_peak_season)
        
        # Interaction features
        country_type_interaction = country_id * attraction_type_id
        features.append(country_type_interaction)
        
        rating_type_interaction = rating * attraction_type_id
        features.append(rating_type_interaction)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(features_scaled)[0]
        
        # Convert back to original label (0-4 to 1-5)
        original_prediction = self.label_mapping.get(prediction, prediction)
        
        return int(original_prediction)
    
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
            'VisitYear', 'VisitMonth', 'Rating', 'ContinentId', 'RegionId', 
            'CountryId', 'CityId', 'AttractionTypeId', 'AttractionCountryId'
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
    predictor = VisitModePredictor()
    best_model = predictor.train_all_models()
    
    # Example prediction
    sample_features = {
        'VisitYear': 2023,
        'VisitMonth': 7,
        'Rating': 4,
        'ContinentId': 1,
        'RegionId': 1,
        'CountryId': 1,
        'CityId': 1,
        'AttractionTypeId': 13,  # Beaches
        'AttractionCountryId': 1
    }
    
    predicted_mode = predictor.predict_visit_mode(sample_features, best_model)
    mode_names = {1: 'Business', 2: 'Couples', 3: 'Family', 4: 'Friends', 5: 'Solo'}
    predicted_mode_name = mode_names.get(predicted_mode, f'Mode {predicted_mode}')
    
    print(f"\nPredicted visit mode: {predicted_mode} ({predicted_mode_name})")
    
    # Show feature importance
    importance = predictor.get_feature_importance('RandomForest')
    print(f"\nFeature Importance:")
    print(importance.head(10))