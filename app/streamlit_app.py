import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.models.regression import RatingPredictor
from src.models.classification import VisitModePredictor
from src.models.recommendation import RecommendationSystem
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced UI styling
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling with animation */
    .main-header {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: fadeInDown 1s ease-out;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.25);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Custom selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize models (with caching)
@st.cache_resource
def load_models():
    """Load all machine learning models."""
    try:
        # Initialize all models
        rating_predictor = RatingPredictor()
        rating_predictor.train_all_models()
        
        visit_mode_predictor = VisitModePredictor()
        visit_mode_predictor.train_all_models()
        
        recommendation_system = RecommendationSystem()
        
        return rating_predictor, visit_mode_predictor, recommendation_system
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        st.error(f"Error loading models. Please check the data files. Details: {str(e)}")
        return None, None, None

# Enhanced loading experience
with st.spinner("🚀 Initializing Tourism Analytics Engine..."):
    rating_pred, visit_pred, rec_system = load_models()
    time.sleep(1)  # Add slight delay for better UX

# Enhanced main header with animation
st.markdown('''
<div class="main-header pulse">
    <h1 style="color: white; font-size: 3rem; margin-bottom: 1rem;">🌍 Tourism Experience Analytics</h1>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
        Advanced AI-powered insights for tourism data with predictive analytics and personalized recommendations
    </p>
    <div style="margin-top: 1rem; font-size: 0.9rem; color: rgba(255,255,255,0.7);">
        Last updated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
    </div>
</div>
'''.format(datetime=datetime.now()), unsafe_allow_html=True)

# Enhanced sidebar with custom styling
st.sidebar.markdown("""
<div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
    <h2 style="color: white; text-align: center; margin-bottom: 1rem;">🧭 Navigation</h2>
</div>
""", unsafe_allow_html=True)

# Enhanced navigation with icons
navigation_options = {
    "📊 Overview": "Overview",
    "📈 Data Insights": "Data Insights", 
    "🔮 Predictions": "Predictions",
    "🎯 Recommendations": "Recommendations",
    "ℹ️ About": "About"
}

selected_nav = st.sidebar.radio(
    "Choose a section:",
    list(navigation_options.keys()),
    index=0
)

page = navigation_options[selected_nav]

if page == "Overview":
    st.header("📊 Project Overview")
    
    # Display basic statistics
    if rating_pred and rating_pred.df is not None:
        df = rating_pred.df
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            st.metric("Unique Users", f"{df['UserId'].nunique():,}")
        with col3:
            st.metric("Attractions", f"{df['AttractionId'].nunique():,}")
        with col4:
            st.metric("Average Rating", f"{df['Rating'].mean():.2f}")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
    else:
        st.warning("Data not available. Please ensure data files are in the correct location.")

elif page == "Data Insights":
    st.header("📈 Data Insights")
    
    if rating_pred and rating_pred.df is not None:
        df = rating_pred.df
        
        # Rating distribution
        st.subheader("Rating Distribution")
        rating_counts = df['Rating'].value_counts().sort_index()
        fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                    labels={'x': 'Rating', 'y': 'Count'},
                    title='Distribution of Attraction Ratings')
        st.plotly_chart(fig, use_container_width=True)
        
        # Visit mode distribution
        if 'VisitModeName' in df.columns:
            st.subheader("Visit Mode Distribution")
            mode_counts = df['VisitModeName'].value_counts()
            fig = px.pie(values=mode_counts.values, names=mode_counts.index,
                        title='Distribution of Visit Modes')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top attractions
        st.subheader("Most Popular Attractions")
        top_attractions = df['Attraction'].value_counts().head(10)
        fig = px.bar(x=top_attractions.values, y=top_attractions.index,
                    orientation='h',
                    labels={'x': 'Number of Visits', 'y': 'Attraction'},
                    title='Top 10 Most Visited Attractions')
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating by visit mode
        if 'VisitModeName' in df.columns:
            st.subheader("Average Rating by Visit Mode")
            avg_rating_by_mode = df.groupby('VisitModeName')['Rating'].mean().sort_values(ascending=True)
            fig = px.bar(x=avg_rating_by_mode.values, y=avg_rating_by_mode.index,
                        orientation='h',
                        labels={'x': 'Average Rating', 'y': 'Visit Mode'},
                        title='Average Rating by Visit Mode')
            st.plotly_chart(fig, use_container_width=True)
        
        # User distribution by continent
        if 'Continent' in df.columns:
            st.subheader("User Distribution by Continent")
            continent_counts = df['Continent'].value_counts()
            fig = px.pie(values=continent_counts.values, names=continent_counts.index,
                        title='User Distribution by Continent')
            st.plotly_chart(fig, use_container_width=True)

elif page == "Predictions":
    st.header("🔮 Predictions")
    
    # Create tabs for different predictions
    tab1, tab2 = st.tabs(["_rating Prediction", "Visit Mode Prediction"])
    
    with tab1:
        st.subheader("Attraction Rating Prediction")
        
        if rating_pred is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                visit_year = st.number_input("Visit Year", min_value=2010, max_value=2030, value=2023)
                visit_month = st.selectbox("Visit Month", list(range(1, 13)), index=6)
                continent_id = st.number_input("Continent ID", min_value=0, value=1)
                region_id = st.number_input("Region ID", min_value=0, value=1)
                
            with col2:
                country_id = st.number_input("Country ID", min_value=0, value=1)
                city_id = st.number_input("City ID", min_value=0, value=1)
                attraction_type_id = st.selectbox("Attraction Type", 
                                                [2, 10, 13, 19, 34, 44, 45, 61, 63, 64, 72, 76, 82, 84, 91, 92, 93],
                                                format_func=lambda x: {2: "Ancient Ruins", 10: "Ballets", 13: "Beaches", 
                                                                     19: "Caverns & Caves", 34: "Flea & Street Markets",
                                                                     44: "Historic Sites", 45: "History Museums", 
                                                                     61: "National Parks", 63: "Nature & Wildlife Areas",
                                                                     64: "Neighborhoods", 72: "Points of Interest & Landmarks",
                                                                     76: "Religious Sites", 82: "Spas", 84: "Speciality Museums",
                                                                     91: "Volcanos", 92: "Water Parks", 93: "Waterfalls"}[x])
                visit_mode = st.selectbox("Visit Mode", [1, 2, 3, 4, 5],
                                        format_func=lambda x: {1: "Business", 2: "Couples", 3: "Family", 4: "Friends", 5: "Solo"}.get(x, f"Mode {x}"))
            
            if st.button("Predict Rating"):
                # Prepare features
                features = {
                    'VisitYear': visit_year,
                    'VisitMonth': visit_month,
                    'ContinentId': continent_id,
                    'RegionId': region_id,
                    'CountryId': country_id,
                    'CityId': city_id,
                    'AttractionTypeId': attraction_type_id,
                    'VisitMode': visit_mode
                }
                
                # Make prediction
                try:
                    predicted_rating = rating_pred.predict_rating(features)
                    st.success(f"Predicted Rating: {predicted_rating:.2f}")
                    
                    # Show confidence interval
                    st.info(f"Expected rating range: {max(1, predicted_rating-0.5):.1f} - {min(5, predicted_rating+0.5):.1f}")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        else:
            st.warning("Rating prediction model not available")
    
    with tab2:
        st.subheader("Visit Mode Prediction")
        
        if visit_pred is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                visit_year = st.number_input("Visit Year", min_value=2010, max_value=2030, value=2023, key="mode_year")
                visit_month = st.selectbox("Visit Month", list(range(1, 13)), index=6, key="mode_month")
                rating = st.slider("Expected Rating", 1.0, 5.0, 3.0, 0.1, key="mode_rating")
                continent_id = st.number_input("Continent ID", min_value=0, value=1, key="mode_continent")
                
            with col2:
                region_id = st.number_input("Region ID", min_value=0, value=1, key="mode_region")
                country_id = st.number_input("Country ID", min_value=0, value=1, key="mode_country")
                city_id = st.number_input("City ID", min_value=0, value=1, key="mode_city")
                attraction_type_id = st.selectbox("Attraction Type", 
                                                [2, 10, 13, 19, 34, 44, 45, 61, 63, 64, 72, 76, 82, 84, 91, 92, 93],
                                                key="mode_type",
                                                format_func=lambda x: {2: "Ancient Ruins", 10: "Ballets", 13: "Beaches", 
                                                                     19: "Caverns & Caves", 34: "Flea & Street Markets",
                                                                     44: "Historic Sites", 45: "History Museums", 
                                                                     61: "National Parks", 63: "Nature & Wildlife Areas",
                                                                     64: "Neighborhoods", 72: "Points of Interest & Landmarks",
                                                                     76: "Religious Sites", 82: "Spas", 84: "Speciality Museums",
                                                                     91: "Volcanos", 92: "Water Parks", 93: "Waterfalls"}.get(x, f"Attraction Type {x}"))
            
            if st.button("Predict Visit Mode"):
                # Prepare features
                features = {
                    'VisitYear': visit_year,
                    'VisitMonth': visit_month,
                    'Rating': rating,
                    'ContinentId': continent_id,
                    'RegionId': region_id,
                    'CountryId': country_id,
                    'CityId': city_id,
                    'AttractionTypeId': attraction_type_id,
                    'AttractionCountryId': country_id
                }
                
                # Make prediction
                try:
                    predicted_mode = visit_pred.predict_visit_mode(features)
                    mode_names = {0: "Unknown", 1: "Business", 2: "Couples", 3: "Family", 4: "Friends", 5: "Solo"}
                    predicted_mode_name = mode_names.get(predicted_mode, f"Mode {predicted_mode}")
                    
                    st.success(f"Predicted Visit Mode: {predicted_mode} ({predicted_mode_name})")
                    
                    # Show probabilities if available
                    st.info("This prediction is based on historical patterns and visitor preferences.")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.error(f"Debug: Features used - {features}")
        else:
            st.warning("Visit mode prediction model not available")

elif page == "Recommendations":
    st.header("🎯 Personalized Recommendations")
    
    if rec_system is not None:
        # User input for recommendations
        st.subheader("Get Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.number_input("User ID", min_value=1, value=14)
            rec_method = st.selectbox("Recommendation Method", 
                                    ["hybrid", "collaborative", "content", "popular"],
                                    format_func=lambda x: {
                                        "hybrid": "Hybrid (Best Overall)",
                                        "collaborative": "Collaborative Filtering",
                                        "content": "Content-Based",
                                        "popular": "Popular Attractions"
                                    }[x])
        
        with col2:
            num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)
        
        if st.button("Get Recommendations"):
            try:
                # Get recommendations
                recommendations = rec_system.get_recommendations_for_user(
                    user_id, num_recommendations, rec_method
                )
                
                if not recommendations.empty:
                    st.subheader(f"Top {num_recommendations} Recommendations for User {user_id}")
                    
                    # Display recommendations in a nice format
                    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
                        with st.expander(f"{i}. {rec['Attraction']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Type:** {rec['AttractionType']}")
                                st.write(f"**Location:** {rec['City']}, {rec['Country']}")
                            
                            with col2:
                                if 'HybridScore' in rec:
                                    st.write(f"**Score:** {rec['HybridScore']:.3f}")
                                elif 'AvgRating' in rec:
                                    st.write(f"**Average Rating:** {rec['AvgRating']:.2f}")
                                    st.write(f"**Visit Count:** {rec['VisitCount']}")
                                elif 'SimilarityScore' in rec:
                                    st.write(f"**Similarity:** {rec['SimilarityScore']:.3f}")
                            
                            # Show additional details if available
                            if 'PredictedRating' in rec:
                                st.progress(rec['PredictedRating'] / 5.0)
                                st.caption(f"Predicted Rating: {rec['PredictedRating']:.2f}/5.0")
                
                else:
                    st.warning("No recommendations available for this user.")
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")
                st.info("Try using a different User ID or recommendation method.")
    else:
        st.warning("Recommendation system not available")

elif page == "About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## Tourism Experience Analytics
    
    This project analyzes tourism data to provide insights and predictions for:
    
    ### 🔍 Features:
    - **Data Analysis**: Comprehensive exploratory data analysis with visualizations
    - **Rating Prediction**: Predict attraction ratings based on various factors
    - **Visit Mode Prediction**: Predict the most likely visit mode (Business, Couples, Family, etc.)
    - **Recommendation System**: Personalized attraction recommendations using hybrid filtering
    - **Interactive Dashboard**: User-friendly web interface for exploring insights
    
    ### 📊 Data Sources:
    The system analyzes multiple tourism datasets including:
    - Transaction data (visits, ratings)
    - User demographics and preferences
    - Attraction information and types
    - Geographic data (cities, countries, regions, continents)
    - Visit mode classifications
    
    ### 🧠 Machine Learning Models:
    - **Regression**: Linear Regression, Random Forest, XGBoost for rating prediction
    - **Classification**: Random Forest, XGBoost for visit mode prediction
    - **Recommendation**: Collaborative filtering, content-based filtering, and hybrid approach
    
    ### 🎯 Business Applications:
    - Personalized travel recommendations
    - Visitor behavior analysis
    - Tourism market insights
    - Attraction performance evaluation
    - Marketing strategy optimization
    
    ### 🛠️ Technologies Used:
    - Python, Pandas, NumPy
    - Scikit-learn, XGBoost
    - Streamlit for web interface
    - Plotly for interactive visualizations
    - Matplotlib, Seaborn for static plots
    
    ### 📈 Model Performance:
    - Rating prediction: R² scores around 0.03-0.05 (challenging due to data complexity)
    - Visit mode prediction: Accuracy ~70-80% with F1 scores around 0.7-0.8
    - Recommendation system: Hybrid approach combining multiple methods
    
    ### 🚀 Future Improvements:
    - Deep learning models for better predictions
    - Real-time data integration
    - Advanced natural language processing for reviews
    - Enhanced collaborative filtering with temporal dynamics
    - Mobile application development
    """)
    
    # Show model performance if available
    if rating_pred and visit_pred:
        st.subheader("Model Performance Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Rating Prediction Models:**")
            for model_name, metrics in rating_pred.results.items():
                st.write(f"- {model_name}: R² = {metrics['test_r2']:.3f}, RMSE = {metrics['test_rmse']:.3f}")
        
        with col2:
            st.write("**Visit Mode Prediction Models:**")
            for model_name, metrics in visit_pred.results.items():
                st.write(f"- {model_name}: Accuracy = {metrics['test_accuracy']:.3f}, F1 = {metrics['test_f1']:.3f}")

# Footer
st.markdown("---")
st.markdown("Developed for Tourism Analytics")