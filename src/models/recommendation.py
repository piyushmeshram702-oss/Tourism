import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationSystem:
    """Tourism attraction recommendation system."""
    
    def __init__(self, data_path: str = "data/processed/merged_tourism_data.csv"):
        """
        Initialize recommendation system.
        
        Args:
            data_path (str): Path to merged data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.user_item_matrix = None
        self.attraction_features = None
        self.cosine_sim = None
        self.attraction_indices = None
        
        self.load_data()
        self.build_user_item_matrix()
        self.build_content_features()
    
    def load_data(self):
        """Load the merged dataset."""
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data: {self.df.shape}")
        else:
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def build_user_item_matrix(self):
        """Build user-item rating matrix for collaborative filtering."""
        # Create user-item matrix
        self.user_item_matrix = self.df.pivot_table(
            index='UserId',
            columns='AttractionId',
            values='Rating',
            aggfunc='mean'
        ).fillna(0)
        
        logger.info(f"User-item matrix shape: {self.user_item_matrix.shape}")
        
        # Calculate item similarity based on user ratings
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
    
    def build_content_features(self):
        """Build content-based features for attractions."""
        # Create feature strings for each attraction
        attraction_features = []
        
        for idx, row in self.df.drop_duplicates('AttractionId').iterrows():
            features = []
            
            # Add attraction type
            if pd.notna(row['AttractionType']):
                features.append(str(row['AttractionType']))
            
            # Add location information
            if pd.notna(row['AttractionCountryId']):
                features.append(str(row['AttractionCountryId']))
            
            if pd.notna(row['AttractionCityName']):
                features.append(str(row['AttractionCityName']))
            
            # Combine features
            feature_string = ' '.join(features)
            attraction_features.append({
                'AttractionId': row['AttractionId'],
                'Attraction': row['Attraction'],
                'features': feature_string
            })
        
        self.attraction_features = pd.DataFrame(attraction_features)
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.attraction_features['features'])
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.attraction_indices = pd.Series(
            self.attraction_features.index, 
            index=self.attraction_features['AttractionId']
        ).drop_duplicates()
        
        logger.info(f"Content-based features built for {len(self.attraction_features)} attractions")
    
    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> pd.DataFrame:
        """
        Get recommendations using collaborative filtering.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            
        Returns:
            pd.DataFrame: Recommended attractions with scores
        """
        if user_id not in self.user_item_matrix.index:
            # New user - return popular attractions
            logger.warning(f"User {user_id} not found, returning popular attractions")
            return self.get_popular_attractions(n_recommendations)
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Get attractions the user hasn't rated
        unrated_attractions = user_ratings[user_ratings == 0].index
        
        if len(unrated_attractions) == 0:
            # User has rated everything - return based on highest rated
            rated_attractions = user_ratings[user_ratings > 0].sort_values(ascending=False)
            top_attractions = rated_attractions.head(n_recommendations).index.tolist()
        else:
            # Calculate predicted ratings for unrated attractions
            predicted_ratings = {}
            
            for attraction_id in unrated_attractions:
                # Get similar attractions that the user has rated
                if attraction_id in self.item_similarity_df.index:
                    similarities = self.item_similarity_df[attraction_id]
                    user_rated_similar = similarities[user_ratings[user_ratings > 0].index]
                    
                    if len(user_rated_similar) > 0:
                        # Weighted average of similar attractions' ratings
                        weighted_sum = (user_rated_similar * user_ratings[user_rated_similar.index]).sum()
                        similarity_sum = user_rated_similar.sum()
                        
                        if similarity_sum > 0:
                            predicted_ratings[attraction_id] = weighted_sum / similarity_sum
                        else:
                            predicted_ratings[attraction_id] = 0
                    else:
                        predicted_ratings[attraction_id] = 0
                else:
                    predicted_ratings[attraction_id] = 0
            
            # Sort by predicted rating
            top_attractions = sorted(predicted_ratings.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:n_recommendations]
            top_attractions = [attraction_id for attraction_id, score in top_attractions]
        
        # Get attraction details
        recommendations = []
        for attraction_id in top_attractions:
            attraction_info = self.df[self.df['AttractionId'] == attraction_id].iloc[0]
            recommendations.append({
                'AttractionId': attraction_id,
                'Attraction': attraction_info['Attraction'],
                'AttractionType': attraction_info['AttractionType'],
                'Country': attraction_info['AttractionCountryId'],
                'City': attraction_info['AttractionCityName'],
                'PredictedRating': predicted_ratings.get(attraction_id, 0) if 'predicted_ratings' in locals() else 0
            })
        
        return pd.DataFrame(recommendations)
    
    def get_content_based_recommendations(self, attraction_id: int, n_recommendations: int = 5) -> pd.DataFrame:
        """
        Get recommendations using content-based filtering.
        
        Args:
            attraction_id (int): Attraction ID to base recommendations on
            n_recommendations (int): Number of recommendations
            
        Returns:
            pd.DataFrame: Recommended attractions with similarity scores
        """
        if attraction_id not in self.attraction_indices.index:
            logger.warning(f"Attraction {attraction_id} not found")
            return pd.DataFrame()
        
        # Get the index of the attraction
        idx = self.attraction_indices[attraction_id]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar attractions (excluding the input attraction)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get attraction IDs
        attraction_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Get attraction details
        recommendations = []
        for i, (attr_idx, similarity) in enumerate(zip(attraction_indices, similarity_scores)):
            attraction_row = self.attraction_features.iloc[attr_idx]
            original_info = self.df[self.df['AttractionId'] == attraction_row['AttractionId']].iloc[0]
            
            recommendations.append({
                'AttractionId': attraction_row['AttractionId'],
                'Attraction': attraction_row['Attraction'],
                'AttractionType': original_info['AttractionType'],
                'Country': original_info['AttractionCountryId'],
                'City': original_info['AttractionCityName'],
                'SimilarityScore': similarity
            })
        
        return pd.DataFrame(recommendations)
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5) -> pd.DataFrame:
        """
        Get hybrid recommendations combining collaborative and content-based filtering.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            
        Returns:
            pd.DataFrame: Recommended attractions
        """
        # Get collaborative recommendations
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
        
        if collab_recs.empty:
            return self.get_popular_attractions(n_recommendations)
        
        # For each collaborative recommendation, get similar attractions
        hybrid_scores = {}
        
        for _, rec in collab_recs.iterrows():
            attraction_id = rec['AttractionId']
            
            # Get content-based similar attractions
            content_recs = self.get_content_based_recommendations(attraction_id, 3)
            
            # Add the original attraction
            hybrid_scores[attraction_id] = {
                'attraction': rec['Attraction'],
                'type': rec['AttractionType'],
                'country': rec['Country'],
                'city': rec['City'],
                'score': rec.get('PredictedRating', 1)  # Default score if not available
            }
            
            # Add similar attractions with reduced scores
            for _, content_rec in content_recs.iterrows():
                similar_id = content_rec['AttractionId']
                if similar_id not in hybrid_scores:
                    hybrid_scores[similar_id] = {
                        'attraction': content_rec['Attraction'],
                        'type': content_rec['AttractionType'],
                        'country': content_rec['Country'],
                        'city': content_rec['City'],
                        'score': rec.get('PredictedRating', 1) * content_rec['SimilarityScore'] * 0.7
                    }
        
        # Sort by score and return top recommendations
        sorted_recommendations = sorted(
            hybrid_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )[:n_recommendations]
        
        # Format results
        recommendations = []
        for attraction_id, info in sorted_recommendations:
            recommendations.append({
                'AttractionId': attraction_id,
                'Attraction': info['attraction'],
                'AttractionType': info['type'],
                'Country': info['country'],
                'City': info['city'],
                'HybridScore': info['score']
            })
        
        return pd.DataFrame(recommendations)
    
    def get_popular_attractions(self, n_recommendations: int = 5) -> pd.DataFrame:
        """
        Get popular attractions based on average ratings and visit counts.
        
        Args:
            n_recommendations (int): Number of recommendations
            
        Returns:
            pd.DataFrame: Popular attractions
        """
        # Calculate popularity score
        attraction_stats = self.df.groupby('AttractionId').agg({
            'Rating': ['mean', 'count'],
            'Attraction': 'first',
            'AttractionType': 'first',
            'AttractionCountryId': 'first',
            'AttractionCityName': 'first'
        }).reset_index()
        
        # Flatten column names
        attraction_stats.columns = ['AttractionId', 'AvgRating', 'VisitCount', 
                                 'Attraction', 'AttractionType', 'Country', 'City']
        
        # Calculate popularity score (weighted average of rating and visit count)
        max_visits = attraction_stats['VisitCount'].max()
        attraction_stats['PopularityScore'] = (
            0.7 * attraction_stats['AvgRating'] + 
            0.3 * (attraction_stats['VisitCount'] / max_visits) * 5
        )
        
        # Get top attractions
        top_attractions = attraction_stats.nlargest(n_recommendations, 'PopularityScore')
        
        recommendations = []
        for _, row in top_attractions.iterrows():
            recommendations.append({
                'AttractionId': row['AttractionId'],
                'Attraction': row['Attraction'],
                'AttractionType': row['AttractionType'],
                'Country': row['Country'],
                'City': row['City'],
                'AvgRating': row['AvgRating'],
                'VisitCount': row['VisitCount'],
                'PopularityScore': row['PopularityScore']
            })
        
        return pd.DataFrame(recommendations)
    
    def get_recommendations_for_user(self, user_id: int, n_recommendations: int = 5, 
                                   method: str = 'hybrid') -> pd.DataFrame:
        """
        Get recommendations for a user using specified method.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            method (str): Recommendation method ('collaborative', 'content', 'hybrid', 'popular')
            
        Returns:
            pd.DataFrame: Recommended attractions
        """
        if method == 'collaborative':
            return self.get_collaborative_recommendations(user_id, n_recommendations)
        elif method == 'content':
            # For content-based, we need a reference attraction
            user_attractions = self.df[self.df['UserId'] == user_id]['AttractionId'].unique()
            if len(user_attractions) > 0:
                return self.get_content_based_recommendations(user_attractions[0], n_recommendations)
            else:
                return self.get_popular_attractions(n_recommendations)
        elif method == 'hybrid':
            return self.get_hybrid_recommendations(user_id, n_recommendations)
        elif method == 'popular':
            return self.get_popular_attractions(n_recommendations)
        else:
            raise ValueError("Method must be 'collaborative', 'content', 'hybrid', or 'popular'")

# Example usage
if __name__ == "__main__":
    # Initialize recommendation system
    rec_system = RecommendationSystem()
    
    # Get recommendations for a user
    user_id = 14  # Example user ID
    recommendations = rec_system.get_recommendations_for_user(user_id, 5, 'hybrid')
    
    print(f"\n=== Recommendations for User {user_id} ===")
    if not recommendations.empty:
        for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
            print(f"{i}. {rec['Attraction']}")
            print(f"   Type: {rec['AttractionType']}")
            print(f"   Location: {rec['City']}, {rec['Country']}")
            if 'HybridScore' in rec:
                print(f"   Score: {rec['HybridScore']:.3f}")
            elif 'AvgRating' in rec:
                print(f"   Avg Rating: {rec['AvgRating']:.2f}")
            print()
    else:
        print("No recommendations available")