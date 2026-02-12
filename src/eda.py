import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TourismEDA:
    """Perform Exploratory Data Analysis on tourism data."""
    
    def __init__(self, data_path: str = "data/processed/merged_tourism_data.csv"):
        """
        Initialize EDA analyzer.
        
        Args:
            data_path (str): Path to merged data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load the merged dataset."""
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data: {self.df.shape}")
        else:
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def basic_statistics(self) -> Dict:
        """
        Generate basic statistics about the dataset.
        
        Returns:
            Dict: Statistical summary
        """
        if self.df is None:
            return {}
            
        stats = {
            'dataset_shape': self.df.shape,
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'rating_stats': self.df['Rating'].describe().to_dict() if 'Rating' in self.df.columns else {},
            'visit_mode_distribution': self.df['VisitMode'].value_counts().to_dict() if 'VisitMode' in self.df.columns else {}
        }
        
        return stats
    
    def plot_rating_distribution(self, save_path: str = "notebooks/rating_distribution.png"):
        """Plot distribution of ratings."""
        if 'Rating' not in self.df.columns:
            logger.warning("Rating column not found")
            return
            
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='Rating')
        plt.title('Distribution of Attraction Ratings', fontsize=16)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        
        # Add value labels on bars
        for i, v in enumerate(self.df['Rating'].value_counts().sort_index()):
            plt.text(i, v + 50, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Rating distribution plot saved to {save_path}")
    
    def plot_visit_mode_distribution(self, save_path: str = "notebooks/visit_mode_distribution.png"):
        """Plot distribution of visit modes."""
        if 'VisitModeName' not in self.df.columns:
            logger.warning("VisitModeName column not found")
            return
            
        plt.figure(figsize=(12, 6))
        mode_counts = self.df['VisitModeName'].value_counts()
        sns.barplot(x=mode_counts.index, y=mode_counts.values)
        plt.title('Distribution of Visit Modes', fontsize=16)
        plt.xlabel('Visit Mode', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(mode_counts.values):
            plt.text(i, v + 50, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Visit mode distribution plot saved to {save_path}")
    
    def plot_top_attractions(self, top_n: int = 10, save_path: str = "notebooks/top_attractions.png"):
        """Plot top attractions by visit count."""
        if 'Attraction' not in self.df.columns:
            logger.warning("Attraction column not found")
            return
            
        plt.figure(figsize=(12, 8))
        attraction_counts = self.df['Attraction'].value_counts().head(top_n)
        sns.barplot(x=attraction_counts.values, y=attraction_counts.index)
        plt.title(f'Top {top_n} Most Visited Attractions', fontsize=16)
        plt.xlabel('Number of Visits', fontsize=12)
        plt.ylabel('Attraction', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Top attractions plot saved to {save_path}")
    
    def plot_rating_by_visit_mode(self, save_path: str = "notebooks/rating_by_visit_mode.png"):
        """Plot rating distribution by visit mode."""
        if 'Rating' not in self.df.columns or 'VisitModeName' not in self.df.columns:
            logger.warning("Required columns not found")
            return
            
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='VisitModeName', y='Rating')
        plt.title('Rating Distribution by Visit Mode', fontsize=16)
        plt.xlabel('Visit Mode', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Rating by visit mode plot saved to {save_path}")
    
    def plot_user_distribution_by_continent(self, save_path: str = "notebooks/user_by_continent.png"):
        """Plot user distribution by continent."""
        if 'Continent' not in self.df.columns:
            logger.warning("Continent column not found")
            return
            
        plt.figure(figsize=(10, 6))
        continent_counts = self.df['Continent'].value_counts()
        plt.pie(continent_counts.values, labels=continent_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('User Distribution by Continent', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"User distribution by continent plot saved to {save_path}")
    
    def plot_correlation_matrix(self, save_path: str = "notebooks/correlation_matrix.png"):
        """Plot correlation matrix of numerical features."""
        # Select numerical columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Remove columns with '_scaled' or '_encoded' suffix for cleaner visualization
        base_columns = [col for col in numeric_df.columns 
                       if not any(suffix in col for suffix in ['_scaled', '_encoded'])]
        
        if len(base_columns) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = numeric_df[base_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix of Numerical Features', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            logger.info(f"Correlation matrix plot saved to {save_path}")
    
    def plot_seasonal_trends(self, save_path: str = "notebooks/seasonal_trends.png"):
        """Plot seasonal trends in tourism."""
        if 'VisitMonth' not in self.df.columns or 'Rating' not in self.df.columns:
            logger.warning("Required columns not found")
            return
            
        # Create season mapping
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        
        self.df['Season'] = self.df['VisitMonth'].map(season_map)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Seasonal visit counts
        season_counts = self.df['Season'].value_counts()
        ax1.pie(season_counts.values, labels=season_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribution of Visits by Season', fontsize=14)
        
        # Average rating by season
        season_rating = self.df.groupby('Season')['Rating'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        bars = ax2.bar(season_rating.index, season_rating.values)
        ax2.set_title('Average Rating by Season', fontsize=14)
        ax2.set_ylabel('Average Rating')
        ax2.set_ylim(0, 5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Seasonal trends plot saved to {save_path}")
    
    def generate_comprehensive_report(self, output_dir: str = "notebooks"):
        """
        Generate all EDA plots and save them.
        
        Args:
            output_dir (str): Directory to save plots
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate all plots
        self.plot_rating_distribution(f"{output_dir}/rating_distribution.png")
        self.plot_visit_mode_distribution(f"{output_dir}/visit_mode_distribution.png")
        self.plot_top_attractions(10, f"{output_dir}/top_attractions.png")
        self.plot_rating_by_visit_mode(f"{output_dir}/rating_by_visit_mode.png")
        self.plot_user_distribution_by_continent(f"{output_dir}/user_by_continent.png")
        self.plot_correlation_matrix(f"{output_dir}/correlation_matrix.png")
        self.plot_seasonal_trends(f"{output_dir}/seasonal_trends.png")
        
        # Print basic statistics
        stats = self.basic_statistics()
        print("\n=== EDA REPORT ===")
        print(f"Dataset Shape: {stats['dataset_shape']}")
        print(f"Numeric Columns: {len(stats['numeric_columns'])}")
        print(f"Categorical Columns: {len(stats['categorical_columns'])}")
        print(f"Missing Values: {sum(stats['missing_values'].values())}")
        if stats['rating_stats']:
            print(f"Rating Statistics: Mean={stats['rating_stats']['mean']:.2f}, Std={stats['rating_stats']['std']:.2f}")
        print("==================\n")

# Example usage
if __name__ == "__main__":
    # Initialize and run EDA
    eda = TourismEDA()
    eda.generate_comprehensive_report()