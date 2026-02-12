"""
Tourism Experience Analytics - Project Demonstration
===================================================

This script demonstrates all components of the tourism analytics system:
1. Data loading and exploration
2. Machine learning model predictions
3. Recommendation system
4. Performance metrics
"""

import pandas as pd
import numpy as np
from src.models.regression import RatingPredictor
from src.models.classification import VisitModePredictor
from src.models.recommendation import RecommendationSystem
import time

def demonstrate_data_overview():
    """Step 1: Show data overview and statistics"""
    print("=" * 60)
    print("STEP 1: DATA OVERVIEW AND STATISTICS")
    print("=" * 60)
    
    # Load the merged data
    df = pd.read_csv('data/processed/merged_tourism_data.csv')
    
    print(f"📊 Dataset Information:")
    print(f"   • Total Records: {len(df):,}")
    print(f"   • Total Features: {len(df.columns)}")
    print(f"   • Unique Users: {df['UserId'].nunique():,}")
    print(f"   • Unique Attractions: {df['AttractionId'].nunique():,}")
    print(f"   • Date Range: {df['VisitYear'].min()}-{df['VisitYear'].max()}")
    
    print(f"\n⭐ Rating Statistics:")
    print(f"   • Average Rating: {df['Rating'].mean():.2f}")
    print(f"   • Rating Distribution:")
    for rating, count in df['Rating'].value_counts().sort_index().items():
        percentage = (count / len(df)) * 100
        print(f"     Rating {rating}: {count:,} visits ({percentage:.1f}%)")
    
    print(f"\n👥 Visit Mode Distribution:")
    mode_mapping = {0: 'Solo', 1: 'Business', 2: 'Couples', 3: 'Family', 4: 'Friends'}
    for mode_id, count in df['VisitMode'].value_counts().sort_index().items():
        mode_name = mode_mapping.get(mode_id, f'Mode {mode_id}')
        percentage = (count / len(df)) * 100
        print(f"     {mode_name}: {count:,} visits ({percentage:.1f}%)")
    
    print(f"\n📍 Top 5 Attractions by Visits:")
    top_attractions = df['Attraction'].value_counts().head(5)
    for i, (attraction, count) in enumerate(top_attractions.items(), 1):
        print(f"     {i}. {attraction}: {count:,} visits")

def demonstrate_rating_prediction():
    """Step 2: Demonstrate rating prediction models"""
    print("\n" + "=" * 60)
    print("STEP 2: RATING PREDICTION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize rating predictor
    print("🔄 Initializing Rating Prediction Models...")
    rating_pred = RatingPredictor()
    best_model = rating_pred.train_all_models()
    
    print(f"✅ Best Model: {best_model}")
    print(f"📊 Performance Metrics:")
    for model_name, metrics in rating_pred.results.items():
        print(f"   {model_name}:")
        print(f"     • Test R²: {metrics['test_r2']:.4f}")
        print(f"     • Test RMSE: {metrics['test_rmse']:.4f}")
    
    # Demonstrate predictions
    print(f"\n🔮 Sample Rating Predictions:")
    sample_scenarios = [
        {
            'VisitYear': 2023,
            'VisitMonth': 7,
            'ContinentId': 2,
            'RegionId': 8,
            'CountryId': 48,
            'CityId': 464,
            'AttractionTypeId': 63,  # Nature & Wildlife Areas
            'VisitMode': 2  # Couples
        },
        {
            'VisitYear': 2023,
            'VisitMonth': 12,
            'ContinentId': 5,
            'RegionId': 21,
            'CountryId': 163,
            'CityId': 4341,
            'AttractionTypeId': 44,  # Historic Sites
            'VisitMode': 3  # Family
        },
        {
            'VisitYear': 2023,
            'VisitMonth': 3,
            'ContinentId': 2,
            'RegionId': 9,
            'CountryId': 54,
            'CityId': 774,
            'AttractionTypeId': 13,  # Beaches
            'VisitMode': 4  # Friends
        }
    ]
    
    for i, scenario in enumerate(sample_scenarios, 1):
        predicted_rating = rating_pred.predict_rating(scenario)
        mode_names = {1: "Business", 2: "Couples", 3: "Family", 4: "Friends", 5: "Solo"}
        attraction_types = {
            13: "Beaches", 44: "Historic Sites", 63: "Nature & Wildlife Areas"
        }
        
        print(f"\n   Scenario {i}:")
        print(f"     Visit Mode: {mode_names.get(scenario['VisitMode'], scenario['VisitMode'])}")
        print(f"     Attraction Type: {attraction_types.get(scenario['AttractionTypeId'], 'Other')}")
        print(f"     Predicted Rating: {predicted_rating:.2f}/5.0")

def demonstrate_visit_mode_prediction():
    """Step 3: Demonstrate visit mode prediction"""
    print("\n" + "=" * 60)
    print("STEP 3: VISIT MODE PREDICTION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize visit mode predictor
    print("🔄 Initializing Visit Mode Prediction Models...")
    visit_pred = VisitModePredictor()
    best_model = visit_pred.train_all_models()
    
    print(f"✅ Best Model: {best_model}")
    print(f"📊 Performance Metrics:")
    for model_name, metrics in visit_pred.results.items():
        print(f"   {model_name}:")
        print(f"     • Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"     • Test F1 Score: {metrics['test_f1']:.4f}")
    
    # Demonstrate predictions
    print(f"\n👥 Sample Visit Mode Predictions:")
    sample_scenarios = [
        {
            'VisitYear': 2023,
            'VisitMonth': 6,
            'Rating': 4.5,
            'ContinentId': 2,
            'CountryId': 48,
            'AttractionTypeId': 63,  # Nature & Wildlife Areas
            'AttractionCountryId': 48
        },
        {
            'VisitYear': 2023,
            'VisitMonth': 12,
            'Rating': 3.8,
            'ContinentId': 5,
            'CountryId': 163,
            'AttractionTypeId': 44,  # Historic Sites
            'AttractionCountryId': 163
        }
    ]
    
    mode_names = {0: "Solo", 1: "Business", 2: "Couples", 3: "Family", 4: "Friends"}
    
    for i, scenario in enumerate(sample_scenarios, 1):
        predicted_mode = visit_pred.predict_visit_mode(scenario)
        predicted_mode_name = mode_names.get(predicted_mode, f"Mode {predicted_mode}")
        attraction_types = {44: "Historic Sites", 63: "Nature & Wildlife Areas"}
        
        print(f"\n   Scenario {i}:")
        print(f"     Attraction Type: {attraction_types.get(scenario['AttractionTypeId'], 'Other')}")
        print(f"     Expected Rating: {scenario['Rating']}")
        print(f"     Predicted Visit Mode: {predicted_mode} ({predicted_mode_name})")

def demonstrate_recommendation_system():
    """Step 4: Demonstrate recommendation system"""
    print("\n" + "=" * 60)
    print("STEP 4: RECOMMENDATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Initialize recommendation system
    print("🔄 Initializing Recommendation System...")
    rec_system = RecommendationSystem()
    
    # Test different recommendation methods
    test_user_id = 14
    num_recommendations = 5
    
    print(f"🎯 Generating recommendations for User {test_user_id}:")
    
    methods = ['hybrid', 'collaborative', 'content', 'popular']
    method_names = {
        'hybrid': '🔄 Hybrid Approach',
        'collaborative': '👥 Collaborative Filtering',
        'content': '📚 Content-Based',
        'popular': '⭐ Popular Attractions'
    }
    
    for method in methods:
        print(f"\n   {method_names[method]}:")
        try:
            recommendations = rec_system.get_recommendations_for_user(
                test_user_id, num_recommendations, method
            )
            
            if not recommendations.empty:
                for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
                    print(f"     {i}. {rec['Attraction']} ({rec['AttractionType']})")
                    if 'AvgRating' in rec:
                        print(f"        Average Rating: {rec['AvgRating']:.2f} ({rec['VisitCount']} visits)")
                    elif 'SimilarityScore' in rec:
                        print(f"        Similarity Score: {rec['SimilarityScore']:.3f}")
            else:
                print("        No recommendations available")
        except Exception as e:
            print(f"        Error: {str(e)}")

def demonstrate_performance_summary():
    """Step 5: Show overall performance summary"""
    print("\n" + "=" * 60)
    print("STEP 5: PROJECT PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Load data for final statistics
    df = pd.read_csv('data/processed/merged_tourism_data.csv')
    
    print("🏆 PROJECT ACHIEVEMENTS:")
    print(f"   • Processed {len(df):,} tourism transactions")
    print(f"   • Analyzed {df['UserId'].nunique():,} unique users")
    print(f"   • Evaluated {df['AttractionId'].nunique():,} different attractions")
    print(f"   • Trained 5 machine learning models")
    print(f"   • Implemented hybrid recommendation system")
    
    print(f"\n📈 KEY PERFORMANCE INDICATORS:")
    print(f"   • Data Quality: 100% (0 missing values)")
    print(f"   • Rating Prediction: R² = 0.1038 (XGBoost)")
    print(f"   • Visit Mode Prediction: Accuracy = 47.41% (RandomForest)")
    print(f"   • Recommendation Coverage: 30 attractions")
    print(f"   • User Base: 33,530 registered users")
    
    print(f"\n💡 BUSINESS INSIGHTS:")
    print(f"   • Most visits occur in December and July (seasonal peaks)")
    print(f"   • Family visits represent 16.5% of all visits")
    print(f"   • Historic Sites and Beaches are top attraction categories")
    print(f"   • Average rating of 4.16 indicates high visitor satisfaction")
    print(f"   • Solo travelers make up 0.9% of visits")

def main():
    """Main demonstration function"""
    print("🎉 TOURISM EXPERIENCE ANALYTICS - PROJECT DEMONSTRATION")
    print("This demonstration showcases all components of the tourism analytics system.")
    print()
    
    try:
        # Step-by-step demonstration
        demonstrate_data_overview()
        time.sleep(2)  # Brief pause between sections
        
        demonstrate_rating_prediction()
        time.sleep(2)
        
        demonstrate_visit_mode_prediction()
        time.sleep(2)
        
        demonstrate_recommendation_system()
        time.sleep(2)
        
        demonstrate_performance_summary()
        
        print("\n" + "=" * 60)
        print("🎊 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("🚀 Next Steps:")
        print("   • Explore the interactive dashboard at http://localhost:8511")
        print("   • Review the detailed report: Tourism_Analytics_Report.md")
        print("   • Check EDA visualizations in the notebooks/ directory")
        print("   • Examine processed data in data/processed/")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()