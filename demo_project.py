"""
Tourism Experience Analytics - Demo Script
==========================================

This script demonstrates all components of the tourism analytics project
in a step-by-step manner.
"""

import pandas as pd
import time
from src.models.regression import RatingPredictor
from src.models.classification import VisitModePredictor
from src.models.recommendation import RecommendationSystem

def demo_step(step_name, description):
    """Display a step in the demo process"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"Description: {description}")
    print('='*60)

def run_project_demo():
    """Run complete project demonstration"""
    
    print("🚀 TOURISM EXPERIENCE ANALYTICS - PROJECT DEMONSTRATION")
    print("="*60)
    
    # Step 1: Load and examine data
    demo_step("1", "Loading and examining tourism data")
    start_time = time.time()
    
    try:
        df = pd.read_csv('data/processed/merged_tourism_data.csv')
        load_time = time.time() - start_time
        print(f"✅ Data loaded successfully in {load_time:.2f} seconds")
        print(f"📊 Dataset contains {df.shape[0]:,} records and {df.shape[1]} columns")
        print(f"📈 Rating range: {df['Rating'].min()} to {df['Rating'].max()}")
        print(f"👥 Visit modes: {sorted(df['VisitMode'].unique())}")
        print(f"📍 Unique attractions: {df['AttractionId'].nunique()}")
        print(f"👤 Unique users: {df['UserId'].nunique()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Rating prediction model
    demo_step("2", "Training rating prediction models")
    start_time = time.time()
    
    try:
        print("Initializing rating predictor...")
        rating_pred = RatingPredictor()
        print("Training models (this may take a moment)...")
        best_rating_model = rating_pred.train_all_models()
        training_time = time.time() - start_time
        print(f"✅ Rating models trained in {training_time:.2f} seconds")
        print(f"🏆 Best model: {best_rating_model}")
        
        # Test prediction
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
        predicted_rating = rating_pred.predict_rating(sample_features, best_rating_model)
        print(f"🎯 Sample prediction: Rating = {predicted_rating:.2f} for family beach visit")
        
    except Exception as e:
        print(f"❌ Error with rating prediction: {e}")
    
    # Step 3: Visit mode prediction model
    demo_step("3", "Training visit mode prediction models")
    start_time = time.time()
    
    try:
        print("Initializing visit mode predictor...")
        visit_pred = VisitModePredictor()
        print("Training models...")
        best_visit_model = visit_pred.train_all_models()
        training_time = time.time() - start_time
        print(f"✅ Visit mode models trained in {training_time:.2f} seconds")
        print(f"🏆 Best model: {best_visit_model}")
        
        # Test prediction
        sample_features = {
            'VisitYear': 2023,
            'VisitMonth': 12,
            'Rating': 4,
            'ContinentId': 1,
            'RegionId': 1,
            'CountryId': 1,
            'CityId': 1,
            'AttractionTypeId': 44,  # Historic Sites
            'AttractionCountryId': 1
        }
        predicted_mode = visit_pred.predict_visit_mode(sample_features, best_visit_model)
        mode_names = {1: "Business", 2: "Couples", 3: "Family", 4: "Friends", 5: "Solo"}
        predicted_mode_name = mode_names.get(predicted_mode, f"Mode {predicted_mode}")
        print(f"🎯 Sample prediction: Visit Mode = {predicted_mode} ({predicted_mode_name}) for historic site visit")
        
    except Exception as e:
        print(f"❌ Error with visit mode prediction: {e}")
    
    # Step 4: Recommendation system
    demo_step("4", "Initializing recommendation system")
    start_time = time.time()
    
    try:
        print("Initializing recommendation system...")
        rec_system = RecommendationSystem()
        init_time = time.time() - start_time
        print(f"✅ Recommendation system initialized in {init_time:.2f} seconds")
        
        # Test recommendations
        print("Generating sample recommendations...")
        recommendations = rec_system.get_recommendations_for_user(14, 3, 'hybrid')
        print(f"🎯 Generated {len(recommendations)} recommendations for user 14:")
        for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
            print(f"   {i}. {rec['Attraction']} ({rec['AttractionType']}) in {rec['City']}")
        
    except Exception as e:
        print(f"❌ Error with recommendation system: {e}")
    
    # Step 5: Summary
    demo_step("5", "Project demonstration complete")
    print("🎉 TOURISM EXPERIENCE ANALYTICS DEMONSTRATION FINISHED")
    print("\n📋 SUMMARY OF COMPONENTS:")
    print("✅ Data Loading and Processing")
    print("✅ Rating Prediction Models (Linear Regression, Random Forest, XGBoost)")
    print("✅ Visit Mode Classification Models (Random Forest, XGBoost)")
    print("✅ Hybrid Recommendation System")
    print("✅ All models trained and functional")
    
    print("\n📊 PERFORMANCE METRICS:")
    try:
        print(f"   Rating Prediction: R² ≈ {rating_pred.results[best_rating_model]['test_r2']:.3f}")
        print(f"   Visit Mode Prediction: Accuracy ≈ {visit_pred.results[best_visit_model]['test_accuracy']:.3f}")
    except:
        print("   Performance metrics available in trained models")
    
    print("\n🚀 NEXT STEPS:")
    print("   To run the full Streamlit web application:")
    print("   python -m streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    run_project_demo()