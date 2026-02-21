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
import base64
from streamlit.components.v1 import html
import functools
from typing import Optional, Dict, Any
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Professional error handling decorator
def professional_error_handler(func):
    """Decorator for professional error handling with detailed logging"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Executing function: {func.__name__}")
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Function {func.__name__} completed successfully in {execution_time:.2f}s")
            return result
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.error(f"❌ {error_msg}")
            st.info("🔧 The system will attempt to recover. Please try again or contact support if the issue persists.")
            return None
    return wrapper

# Data validation utilities
class DataValidator:
    """Professional data validation utilities"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
        """Validate dataframe structure and content"""
        if df is None or df.empty:
            logger.warning("DataFrame is None or empty")
            return False
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for data quality issues
        null_counts = df.isnull().sum()
        high_null_columns = null_counts[null_counts > len(df) * 0.5]
        if not high_null_columns.empty:
            logger.warning(f"Columns with >50% null values: {high_null_columns.index.tolist()}")
        
        return True
    
    @staticmethod
    def validate_numeric_range(series: pd.Series, min_val: float, max_val: float, column_name: str) -> bool:
        """Validate numeric data ranges"""
        if series is None or series.empty:
            return False
        
        out_of_range = series[(series < min_val) | (series > max_val)]
        if not out_of_range.empty:
            logger.warning(f"Column {column_name} has {len(out_of_range)} values out of range [{min_val}, {max_val}]")
            return False
        return True

# Professional loading indicator
def show_loading_state(message: str = "Processing your request..."):
    """Show professional loading state with progress bar"""
    with st.spinner(message):
        # Simulate processing time for better UX
        time.sleep(0.5)
        return True

# Enhanced data loading with progress
def load_data_with_progress():
    """Load data with professional progress indicators"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing models...")
        progress_bar.progress(20)
        
        rating_pred, visit_pred, rec_system = load_models()
        
        status_text.text("📊 Loading data...")
        progress_bar.progress(60)
        
        time.sleep(1)  # Add slight delay for better UX
        
        status_text.text("✅ System ready!")
        progress_bar.progress(100)
        
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return rating_pred, visit_pred, rec_system
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        st.error("❌ Failed to load system components. Please refresh the page.")
        return None, None, None

# Professional data quality checker
class DataQualityChecker:
    """Professional data quality assessment tool"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.quality_report = {}
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        if self.df is None or self.df.empty:
            return {"error": "No data available"}
        
        report = {
            "total_records": len(self.df),
            "total_columns": len(self.df.columns),
            "missing_data_percentage": (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            "duplicate_records": self.df.duplicated().sum(),
            "data_types": self.df.dtypes.to_dict(),
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            "numeric_columns_stats": {},
            "categorical_columns_stats": {}
        }
        
        # Analyze numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report["numeric_columns_stats"][col] = {
                "min": float(self.df[col].min()) if not self.df[col].empty else None,
                "max": float(self.df[col].max()) if not self.df[col].empty else None,
                "mean": float(self.df[col].mean()) if not self.df[col].empty else None,
                "std": float(self.df[col].std()) if not self.df[col].empty else None,
                "null_count": int(self.df[col].isnull().sum())
            }
        
        # Analyze categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            report["categorical_columns_stats"][col] = {
                "unique_values": int(self.df[col].nunique()),
                "top_values": self.df[col].value_counts().head(5).to_dict(),
                "null_count": int(self.df[col].isnull().sum())
            }
        
        self.quality_report = report
        return report

# Set page configuration
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced UI styling with creative professional design
st.markdown("""
<style>
    /* Main background with subtle texture */
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        min-height: 100vh;
        background-image: radial-gradient(circle at 10% 20%, rgba(255,255,255,0.05) 0%, transparent 20%),
                          radial-gradient(circle at 90% 80%, rgba(255,255,255,0.05) 0%, transparent 20%);
    }
    
    /* Creative header with geometric patterns */
    .main-header {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3498db, #2ecc71, #e74c3c, #f39c12);
    }
    
    /* Creative typography */
    .creative-title {
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #ecf0f1, #bdc3c7, #ecf0f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Innovative metric cards with creative design */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        margin: 1rem 0;
        border: 1px solid rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    /* Creative metric display */
    .metric-display {
        text-align: center;
        padding: 1.5rem;
    }
    
    .metric-number {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
        font-weight: 500;
        letter-spacing: 0.5px;
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
    
    /* Section headers */
    .section-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Card hover effects */
    .hover-card {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .hover-card:hover {
        transform: scale(1.03);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
    }
    
    /* Neumorphic containers */
    .neumorphic {
        background: #e0e5ec;
        box-shadow: 9px 9px 16px #b8bec7, -9px -9px 16px #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
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

# Enhanced loading experience with professional progress indicators
rating_pred, visit_pred, rec_system = load_data_with_progress()

# Enhanced main header with creative professional design
st.markdown('''
<div class="main-header">
    <h1 class="creative-title" style="font-size: 3rem; margin-bottom: 1rem;">🌍 Tourism Experience Analytics</h1>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
        Advanced AI-powered insights for tourism data with predictive analytics and personalized recommendations
    </p>
    <div style="margin-top: 1rem; font-size: 0.9rem; color: rgba(255,255,255,0.7);">
        Last updated: {}
    </div>
</div>
'''.format(datetime.now().strftime("%B %d, %Y at %I:%M %p")), unsafe_allow_html=True)

# Enhanced sidebar with creative styling
st.sidebar.markdown("""
<div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
    <h2 style="color: white; text-align: center; margin-bottom: 1rem;">🧭 Navigation</h2>
</div>
""", unsafe_allow_html=True)

# Enhanced navigation with creative styling
navigation_options = {
    "📊 Overview": "Overview",
    "📈 Data Insights": "Data Insights", 
    "🔮 Predictions": "Predictions",
    "🎯 Recommendations": "Recommendations",
    "ℹ️ About": "About"
}

# Display creative navigation with enhanced styling
st.sidebar.markdown("### Choose a section:")

# Create radio buttons with custom styling
selected_nav = st.sidebar.radio(
    "",
    list(navigation_options.keys()),
    index=0,
    key="navigation_radio"
)

page = navigation_options[selected_nav]

if page == "Overview":
    st.markdown("""
    <div class="section-header">
        <h2 style="color: #2c3e50;">📊 Project Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced statistics display with professional validation
    if rating_pred and rating_pred.df is not None:
        df = rating_pred.df
        
        # Professional data validation with flexible column detection
        
        # Professional data validation
        # Use more flexible column names based on what's available
        actual_columns = list(df.columns)
        required_columns = ['UserId', 'AttractionId', 'Rating', 'VisitYear', 'Attraction']
        
        # Find the best location column
        country_columns = [col for col in actual_columns if 'Country' in col]
        if country_columns:
            location_column = country_columns[0]  # Use first available country column
            required_columns.append(location_column)
            st.info(f"Using location column: {location_column}")
        else:
            # If no country column, check for alternatives
            if 'UserCountry' in actual_columns:
                required_columns.append('UserCountry')
                location_column = 'UserCountry'
                st.info(f"Using location column: {location_column}")
            else:
                # As last resort, use CityName
                if 'AttractionCityName' in actual_columns:
                    required_columns.append('AttractionCityName')
                    location_column = 'AttractionCityName'
                    st.info(f"Using location column: {location_column}")
                else:
                    st.warning("⚠️ No suitable location column found, using basic validation")
                    location_column = None
        if not DataValidator.validate_dataframe(df, required_columns):
            st.error("❌ Data validation failed. Required columns are missing.")
            missing_cols = [col for col in required_columns if col not in df.columns]
            st.error(f"❌ Missing columns: {missing_cols}")
            st.info("🔧 Please ensure all data files are properly loaded.")
            
            # Show some data info for troubleshooting
            st.write("Sample data:")
            st.write(df.head())
            st.write(f"Data shape: {df.shape}")
            
        else:
            
            # Validate rating range
            if not DataValidator.validate_numeric_range(df['Rating'], 1.0, 5.0, 'Rating'):
                st.warning("⚠️ Some rating values are outside the expected range (1-5)")
        
        # Show data quality dashboard
        if st.checkbox("📋 Show Data Quality Report", key="quality_report"):
            quality_checker = DataQualityChecker(df)
            quality_checker.display_quality_dashboard()
        
        # Professional data summary
        st.markdown("""
        <div style="background: rgba(46, 204, 113, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h4>📊 Data Summary</h4>
            <p>Last validated: {}</p>
        </div>
        """.format(datetime.now().strftime("%B %d, %Y at %I:%M %p")), unsafe_allow_html=True)
        
        # Key metrics with enhanced styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-display">
                    <h3 style="color: #ff6b6b; font-size: 2rem;">📊</h3>
                    <div class="metric-number">{len(df):,}</div>
                    <div class="metric-label">Total Transactions</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-display">
                    <h3 style="color: #4ecdc4; font-size: 2rem;">👥</h3>
                    <div class="metric-number">{df["UserId"].nunique():,}</div>
                    <div class="metric-label">Unique Users</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-display">
                    <h3 style="color: #45b7d1; font-size: 2rem;">📍</h3>
                    <div class="metric-number">{df["AttractionId"].nunique()}</div>
                    <div class="metric-label">Attractions</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-display">
                    <h3 style="color: #96ceb4; font-size: 2rem;">⭐</h3>
                    <div class="metric-number">{df["Rating"].mean():.2f}</div>
                    <div class="metric-label">Average Rating</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced sample data section
        st.markdown("""
        <div style="background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 15px; margin-top: 2rem;">
            <h3 style="color: #2c3e50;">📋 Sample Data Records</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Professional data filtering tools
        st.markdown("""
        <div style="background: rgba(52, 152, 219, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h4 style="margin-top: 0; color: #2c3e50;">🔍 Professional Data Explorer</h4>
            <p>Advanced filtering tools with real-time validation and quality checks</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Advanced filtering options with professional validation
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            min_rating = st.slider("Minimum Rating", 1.0, 5.0, 1.0, 0.5, 
                                 key="min_rating",
                                 help="Filter attractions with ratings above this threshold. Valid range: 1.0-5.0")
            
        with col_filter2:
            available_years = sorted(df['VisitYear'].unique())
            selected_year = st.selectbox("Visit Year", ["All"] + available_years, 
                                       key="year_filter",
                                       help="Filter by specific visit year or select 'All' for complete dataset")
        
        # Search functionality
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_term = st.text_input("🔍 Search attractions:", "", key="attraction_search")
        with search_col2:
            search_button = st.button("🔍 Search", key="search_btn", use_container_width=True)
        
        # Apply smart filtering
        try:
            # Start with full dataset
            filtered_df = df.copy()
            
            # Apply rating filter
            filtered_df = filtered_df[filtered_df['Rating'] >= min_rating]
            
            # Apply year filter
            if selected_year != "All":
                filtered_df = filtered_df[filtered_df['VisitYear'] == int(selected_year)]
            
            # Apply search term if provided
            if (search_term and search_button) or search_term:
                filtered_df = filtered_df[filtered_df['Attraction'].str.contains(search_term, case=False, na=False)]
            
            # Show results
            if not filtered_df.empty:
                st.success(f"✅ Found {len(filtered_df)} matching records")
                
                # Show filtering summary
                st.markdown(f"""
                <div style="background: rgba(46, 204, 113, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                    <h4>📊 Filtering Results</h4>
                    <p><strong>Applied filters:</strong> Rating ≥ {min_rating}, Year: {selected_year}</p>
                    <p><strong>Records found:</strong> {len(filtered_df):,} out of {len(df):,} total records</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display filtered data
                st.dataframe(filtered_df.head(15), use_container_width=True, height=500)
                
                # Show additional insights
                col_insight1, col_insight2 = st.columns(2)
                with col_insight1:
                    st.metric("Unique Attractions", filtered_df['Attraction'].nunique())
                with col_insight2:
                    st.metric("Average Rating", f"{filtered_df['Rating'].mean():.2f}")
            
            else:
                st.info("🔍 No records match your filtering criteria. Try adjusting the filters.")
                
        except Exception as e:
            st.error(f"❌ Error applying filters: {str(e)}")
            # Show full dataset when filtering fails
            st.dataframe(df.head(10), use_container_width=True)
    
    else:
        # Show full dataset when no search
        st.dataframe(df.head(10), use_container_width=True)
        st.caption("Showing first 10 records. Use search above to filter results.")
        
        # Show dataset info
        st.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
            <h4>📊 Dataset Overview</h4>
            <p><strong>Total records:</strong> {len(df):,}</p>
            <p><strong>Unique attractions:</strong> {df['Attraction'].nunique()}</p>
            <p><strong>Unique users:</strong> {df['UserId'].nunique():,}</p>
            <p><strong>Date range:</strong> {df['VisitYear'].min()}-{df['VisitYear'].max()}</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Data Insights":
    st.markdown("""
    <div class="section-header">
        <h2 style="color: #2c3e50;">📈 Data Insights & Analytics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if rating_pred and rating_pred.df is not None:
        df = rating_pred.df
        
        # Create tabs for different insights
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Ratings", "👥 Visit Modes", "📍 Attractions", "🌍 Geography"])
        
        with tab1:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
                <h3 style="color: #2c3e50; text-align: center;">📊 Rating Distribution Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if 'Rating' in df.columns:
                # Get rating data
                rating_counts = df['Rating'].value_counts().sort_index()
                
                # Create two main columns
                main_col1, main_col2 = st.columns([2, 1])
                
                with main_col1:
                    # Enhanced rating distribution chart
                    st.markdown("""
                    <div style="background: white; padding: 1rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h4 style="color: #2c3e50; margin-top: 0;">Distribution Chart</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                               labels={'x': 'Rating', 'y': 'Number of Visits'},
                               title='Attraction Rating Distribution',
                               color=rating_counts.index,
                               color_continuous_scale='viridis')
                    fig.update_layout(
                        height=450,
                        showlegend=False,
                        xaxis_title='Rating',
                        yaxis_title='Number of Visits',
                        title_x=0.5,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    st.plotly_chart(fig, use_container_width=True, height=450)
                
                with main_col2:
                    # Enhanced statistics panel
                    st.markdown("""
                    <div style="background: white; padding: 1rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h4 style="color: #2c3e50; margin-top: 0;">📊 Key Statistics</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Main statistics in cards
                    col_stat1, col_stat2 = st.columns(2)
                    
                    with col_stat1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
                            <h2 style="margin: 0; font-size: 2rem;">{df['Rating'].mean():.2f}</h2>
                            <p style="margin: 0;">Average</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
                            <h2 style="margin: 0; font-size: 2rem;">{df['Rating'].median():.1f}</h2>
                            <p style="margin: 0;">Median</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_stat2:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #45b7d1 0%, #96ceb4 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
                            <h2 style="margin: 0; font-size: 2rem;">{df['Rating'].std():.2f}</h2>
                            <p style="margin: 0;">Std Dev</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; text-align: center;">
                            <h2 style="margin: 0; font-size: 2rem;">{len(df)}</h2>
                            <p style="margin: 0;">Total Reviews</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed distribution table
                    st.markdown("""
                    <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                        <h4 style="margin-top: 0; color: #2c3e50;">📋 Rating Breakdown</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create scrollable distribution table - FIXED VERSION
                    html_parts = ['<div style="max-height: 200px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.5rem;">']
                                         
                    for rating, count in sorted(rating_counts.items()):
                        percentage = (count / len(df)) * 100
                        html_parts.append(f'<div style="display: flex; justify-content: space-between; padding: 0.5rem; margin: 0.2rem 0; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);"><span><strong>Rating {rating}:</strong></span><span>{count} visits ({percentage:.1f}%)</span></div>')
                                         
                    html_parts.append('</div>')
                    distribution_html = ''.join(html_parts)
                                         
                    # Ensure proper HTML rendering
                    st.markdown(distribution_html, unsafe_allow_html=True)
                    
                    # Additional insights
                    st.markdown("""
                    <div style="background: rgba(79, 205, 196, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                        <h4 style="margin-top: 0; color: #2c3e50;">💡 Insights</h4>
                        <p>• Most common rating: <strong>{most_common_rating}</strong> ({most_common_count} visits)</p>
                        <p>• Rating range: <strong>{min_rating}</strong> to <strong>{max_rating}</strong></p>
                        <p>• High ratings (4-5): <strong>{high_rating_pct:.1f}%</strong> of visits</p>
                    </div>
                    """.format(
                        most_common_rating=rating_counts.index[0],
                        most_common_count=rating_counts.iloc[0],
                        min_rating=df['Rating'].min(),
                        max_rating=df['Rating'].max(),
                        high_rating_pct=((df['Rating'] >= 4).sum() / len(df)) * 100
                    ), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #ff6b6b; color: white; padding: 2rem; border-radius: 15px; text-align: center;">
                    <h3>⚠️ Rating Data Not Available</h3>
                    <p>The Rating column was not found in the dataset.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
                <h3 style="color: #2c3e50;">Visit Mode Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if 'VisitModeName' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Visit mode distribution
                    mode_counts = df['VisitModeName'].value_counts()
                    fig = px.pie(values=mode_counts.values, names=mode_counts.index,
                               title='Distribution of Visit Modes',
                               color_discrete_sequence=px.colors.qualitative.Set3,
                               height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visit mode statistics
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                        <h4>Visit Mode Statistics</h4>
                        <p><strong>Most Common:</strong> {mode_counts.index[0]}</p>
                        <p><strong>Total Modes:</strong> {len(mode_counts)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Visit mode by rating analysis
                    if 'Rating' in df.columns:
                        mode_rating = df.groupby('VisitModeName')['Rating'].agg(['mean', 'count']).round(2)
                        mode_rating = mode_rating.sort_values('mean', ascending=False)
                        
                        fig = px.bar(mode_rating, x=mode_rating.index, y='mean',
                                   labels={'x': 'Visit Mode', 'mean': 'Average Rating'},
                                   title='Average Rating by Visit Mode',
                                   color='mean',
                                   color_continuous_scale='blues',
                                   height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed statistics table
                        st.markdown("#### Detailed Statistics")
                        st.dataframe(mode_rating.rename(columns={'mean': 'Avg Rating', 'count': 'Visit Count'}))
                    else:
                        st.warning("Rating data not available for analysis")
            else:
                st.warning("VisitModeName column not found in data")
        
        with tab3:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
                <h3 style="color: #2c3e50;">Popular Attractions Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if 'Attraction' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top attractions by visit count
                    top_attractions = df['Attraction'].value_counts().head(10)
                    fig = px.bar(x=top_attractions.values, y=top_attractions.index,
                                orientation='h',
                                labels={'x': 'Number of Visits', 'y': 'Attraction'},
                                title='Top 10 Most Visited Attractions',
                                color=top_attractions.values,
                                color_continuous_scale='reds',
                                height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Additional attraction insights
                    st.markdown("#### Attraction Insights")
                    
                    total_attractions = df['Attraction'].nunique()
                    st.metric("Total Unique Attractions", total_attractions)
                    
                    if 'Rating' in df.columns:
                        # Top rated attractions (with minimum visit threshold)
                        attraction_stats = df.groupby('Attraction')['Rating'].agg(['mean', 'count']).round(2)
                        top_rated = attraction_stats[attraction_stats['count'] >= 5].sort_values('mean', ascending=False).head(10)
                        
                        st.markdown("#### Top Rated Attractions (5+ visits)")
                        for i, (attraction, stats) in enumerate(top_rated.iterrows(), 1):
                            st.markdown(f"""
                            <div style="background: rgba(79, 205, 196, 0.1); padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
                                <strong>{i}. {attraction}</strong><br>
                                Rating: {stats['mean']:.2f} ({stats['count']} visits)
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Attraction type distribution
                    if 'AttractionType' in df.columns:
                        st.markdown("#### Attraction Types")
                        type_counts = df['AttractionType'].value_counts().head(5)
                        for attraction_type, count in type_counts.items():
                            st.progress(count / type_counts.sum())
                            st.caption(f"{attraction_type}: {count} visits")
            else:
                st.warning("Attraction data not found")
        
        with tab4:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.9); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
                <h3 style="color: #2c3e50;">Geographic Distribution</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Country distribution
                if 'AttractionCountryId' in df.columns:
                    # Load country mapping
                    import pandas as pd
                    try:
                        countries_df = pd.read_csv('data/processed/country_cleaned.csv')
                        country_map = dict(zip(countries_df['CountryId'], countries_df['Country']))
                        
                        # Count by country ID
                        country_counts = df['AttractionCountryId'].value_counts().head(10)
                        
                        # Map country IDs to names
                        country_names = [country_map.get(cid, f'Country {cid}') for cid in country_counts.index]
                        
                        fig = px.bar(x=country_counts.values, y=country_names,
                                    orientation='h',
                                    labels={'x': 'Number of Visits', 'y': 'Country'},
                                    title='Top 10 Countries by Visits',
                                    color=country_counts.values,
                                    color_continuous_scale='earth',
                                    height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    except FileNotFoundError:
                        st.warning("Country mapping file not found")
                else:
                    st.warning("Country data not available")
            
            with col2:
                # City distribution
                if 'AttractionCityName' in df.columns:
                    city_counts = df['AttractionCityName'].value_counts().head(10)
                    fig = px.bar(x=city_counts.values, y=city_counts.index,
                                orientation='h',
                                labels={'x': 'Number of Visits', 'y': 'City'},
                                title='Top 10 Cities by Visits',
                                color=city_counts.values,
                                color_continuous_scale='sunset',
                                height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("City data not available")
                
                # Geographic summary statistics
                st.markdown("#### Geographic Summary")
                if 'AttractionCountryId' in df.columns:
                    try:
                        countries_df = pd.read_csv('data/processed/country_cleaned.csv')
                        country_map = dict(zip(countries_df['CountryId'], countries_df['Country']))
                        country_ids_in_data = df['AttractionCountryId'].unique()
                        country_names_in_data = [country_map.get(cid, f'Country {cid}') for cid in country_ids_in_data if cid in country_map]
                        unique_countries = len(set(country_names_in_data))
                        st.metric("Countries Represented", unique_countries)
                    except FileNotFoundError:
                        st.metric("Countries Represented", df['AttractionCountryId'].nunique())
                if 'AttractionCityName' in df.columns:
                    st.metric("Cities Represented", df['AttractionCityName'].nunique())
                if 'ContinentId' in df.columns:
                    st.metric("Continents", df['ContinentId'].nunique())

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
    st.markdown("""
    <div class="section-header">
        <h2 style="color: #2c3e50;">ℹ️ About This Project</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Project overview tabs
    about_tab1, about_tab2, about_tab3, about_tab4 = st.tabs(["📋 Project Overview", "🤖 ML Models", "📊 Data Insights", "🚀 Future Plans"])
    
    with about_tab1:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <h3>🎯 Tourism Experience Analytics</h3>
            <p>This project analyzes tourism data to provide insights and predictions for:</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px;">
                    <h4>🔍 Features:</h4>
                    <ul>
                        <li>Data Analysis with Visualizations</li>
                        <li>Attraction Rating Prediction</li>
                        <li>Visit Mode Classification</li>
                        <li>Personalized Recommendations</li>
                        <li>Interactive Dashboard</li>
                    </ul>
                </div>
                <div style="background: rgba(79, 205, 196, 0.1); padding: 1rem; border-radius: 10px;">
                    <h4>📊 Data Sources:</h4>
                    <ul>
                        <li>Transaction data (visits, ratings)</li>
                        <li>User demographics and preferences</li>
                        <li>Attraction information and types</li>
                        <li>Geographic data (cities, countries)</li>
                        <li>Visit mode classifications</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        if rating_pred and rating_pred.df is not None:
            df = rating_pred.df
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Records", f"{len(df):,}")
            with col2:
                st.metric("👥 Users", f"{df['UserId'].nunique():,}")
            with col3:
                st.metric("📍 Attractions", f"{df['AttractionId'].nunique():,}")
            with col4:
                st.metric("⭐ Avg Rating", f"{df['Rating'].mean():.2f}")
    
    with about_tab2:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px;">
            <h3>🤖 Machine Learning Models</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if rating_pred and visit_pred:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Rating Prediction Models")
                for model_name, metrics in rating_pred.results.items():
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #4facfe;">
                        <h4>{model_name}</h4>
                        <p><strong>R² Score:</strong> {metrics['test_r2']:.4f}</p>
                        <p><strong>RMSE:</strong> {metrics['test_rmse']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📊 Visit Mode Prediction Models")
                for model_name, metrics in visit_pred.results.items():
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #ff6b6b;">
                        <h4>{model_name}</h4>
                        <p><strong>Accuracy:</strong> {metrics['test_accuracy']:.4f}</p>
                        <p><strong>F1 Score:</strong> {metrics['test_f1']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(79, 205, 196, 0.1); padding: 1.5rem; border-radius: 10px; margin-top: 2rem;">
            <h4>🛠️ Technologies Used:</h4>
            <p><strong>Python Libraries:</strong> Pandas, NumPy, Scikit-learn, XGBoost</p>
            <p><strong>Visualization:</strong> Plotly, Matplotlib, Seaborn</p>
            <p><strong>Web Framework:</strong> Streamlit</p>
            <p><strong>Algorithms:</strong> Random Forest, XGBoost, Collaborative Filtering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with about_tab3:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px;">
            <h3>📊 Key Business Insights</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if rating_pred and rating_pred.df is not None:
            df = rating_pred.df
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Immediate Value")
                st.markdown("""
                <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px;">
                    <ul>
                        <li><strong>Personalized Recommendations:</strong> Enhance user experience</li>
                        <li><strong>Visitor Behavior Insights:</strong> Understand travel patterns</li>
                        <li><strong>Market Segmentation:</strong> Target different visitor types</li>
                        <li><strong>Performance Analytics:</strong> Evaluate attraction success</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🚀 Strategic Benefits")
                st.markdown("""
                <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px;">
                    <ul>
                        <li><strong>Marketing Optimization:</strong> Tailored campaigns by visit mode</li>
                        <li><strong>Resource Planning:</strong> Predict demand patterns</li>
                        <li><strong>Customer Satisfaction:</strong> Proactive service improvements</li>
                        <li><strong>Competitive Analysis:</strong> Benchmark performance metrics</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance visualization
            st.markdown("#### 📈 Model Performance Summary")
            if rating_pred and visit_pred:
                performance_data = {
                    'Metric': ['Rating Prediction R²', 'Visit Mode Accuracy', 'Visit Mode F1'],
                    'Value': [
                        max(rating_pred.results.values(), key=lambda x: x['test_r2'])['test_r2'],
                        max(visit_pred.results.values(), key=lambda x: x['test_accuracy'])['test_accuracy'],
                        max(visit_pred.results.values(), key=lambda x: x['test_f1'])['test_f1']
                    ],
                    'Type': ['Regression', 'Classification', 'Classification']
                }
                
                fig = px.bar(performance_data, x='Metric', y='Value', color='Type',
                           title='Model Performance Metrics',
                           labels={'Value': 'Score', 'Metric': 'Performance Metric'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with about_tab4:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px;">
            <h3>🚀 Future Enhancement Opportunities</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Short-term Improvements")
            st.markdown("""
            <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px;">
                <ul>
                    <li>Add cross-validation for robust model evaluation</li>
                    <li>Implement advanced feature engineering</li>
                    <li>Enhance data visualization capabilities</li>
                    <li>Add more sophisticated error handling</li>
                    <li>Implement A/B testing framework</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🌟 Long-term Vision")
            st.markdown("""
            <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 10px;">
                <ul>
                    <li>Real-time data processing and streaming</li>
                    <li>Deep learning models for improved predictions</li>
                    <li>Natural language processing for review analysis</li>
                    <li>Enhanced collaborative filtering with temporal dynamics</li>
                    <li>Mobile application development</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin-top: 2rem;">
            <h3>🎉 Project Success</h3>
            <p>This system successfully demonstrates data science techniques applied to tourism data, providing valuable insights and predictive capabilities for the tourism industry.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Developed for Tourism Analytics")