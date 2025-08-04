"""
Simple training script for the hybrid music genre classifier.
This script will help you get started with training your model.

NOTE: This imports HybridMusicGenreClassifier from the hybrid_model_trainer.py 
file that should be in the same directory (not an external library).
"""

import os
import pandas as pd
import numpy as np
import logging

# Import our custom hybrid model class from the local file
# Make sure hybrid_model_trainer.py is in the same directory!
try:
    from hybrid_model_trainer import HybridMusicGenreClassifier
except ImportError as e:
    print("❌ ERROR: Could not import HybridMusicGenreClassifier")
    print("Make sure 'hybrid_model_trainer.py' is in the same directory!")
    print(f"Import error: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """
    Create sample data for demonstration purposes.
    Replace this with your actual data loading logic.
    """
    logger.info("Creating sample CSV data...")
    
    # Sample genres
    genres = ['rock', 'pop', 'jazz', 'classical', 'electronic', 'hip-hop', 'country', 'blues']
    
    # Create random features (replace with your actual feature extraction)
    np.random.seed(42)
    n_samples_per_genre = 100
    n_features = 33  # Match the feature extraction in your code
    
    data = []
    labels = []
    
    for genre in genres:
        logger.info(f"Creating samples for genre: {genre}")
        for i in range(n_samples_per_genre):
            # Generate random features (this is just for demo - use real features)
            features = np.random.normal(0, 1, n_features)
            
            # Add some genre-specific patterns (simplified)
            if genre == 'rock':
                features[0] += 2  # Higher spectral centroid
                features[29] += 20  # Higher tempo
            elif genre == 'classical':
                features[26:28] += 1  # Higher chroma features
                features[29] -= 10  # Lower tempo
            elif genre == 'electronic':
                features[4:6] += 2  # Higher spectral bandwidth
                features[29] += 30  # Much higher tempo
            elif genre == 'jazz':
                features[26:28] += 2  # Complex chroma features
                features[8:21] += 0.5  # Rich MFCC features
            elif genre == 'pop':
                features[29] += 10  # Moderate tempo increase
                features[0] += 1  # Slightly higher spectral centroid
            elif genre == 'hip-hop':
                features[29] += 15  # Higher tempo
                features[6:8] += 1  # Higher zero-crossing rate
            elif genre == 'country':
                features[26:28] += 0.5  # Moderate chroma features
                features[29] += 5  # Slightly higher tempo
            elif genre == 'blues':
                features[8:21] += 0.3  # Rich MFCC features
                features[29] -= 5  # Slightly lower tempo
            
            # Ensure no infinite or NaN values
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            data.append(features)
            labels.append(genre)
    
    # Convert to numpy arrays first to check data integrity
    data_array = np.array(data, dtype=np.float64)
    labels_array = np.array(labels, dtype=str)
    
    logger.info(f"Created data shape: {data_array.shape}")
    logger.info(f"Data range: min={data_array.min():.2f}, max={data_array.max():.2f}")
    logger.info(f"Labels shape: {labels_array.shape}")
    logger.info(f"Unique labels: {np.unique(labels_array)}")
    
    # Create proper feature names
    feature_names = [
        'spectral_centroid_mean', 'spectral_centroid_std',
        'spectral_rolloff_mean', 'spectral_rolloff_std',
        'spectral_bandwidth_mean', 'spectral_bandwidth_std',
        'zcr_mean', 'zcr_std'
    ]
    
    # Add MFCC features
    for i in range(13):
        feature_names.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])
    
    feature_names.extend(['chroma_mean', 'chroma_std', 'tempo', 'rms_mean', 'rms_std'])
    
    # Verify we have the right number of features
    if len(feature_names) != n_features:
        logger.error(f"Feature name mismatch: expected {n_features}, got {len(feature_names)}")
        # Adjust if needed
        while len(feature_names) < n_features:
            feature_names.append(f'feature_{len(feature_names)}')
        feature_names = feature_names[:n_features]
    
    logger.info(f"Feature names: {feature_names}")
    
    # Create DataFrame
    df = pd.DataFrame(data_array, columns=feature_names)
    df['genre'] = labels_array
    
    # Final validation
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame dtypes: {df.dtypes.unique()}")
    logger.info(f"Any null values: {df.isnull().any().any()}")
    
    return df

def main():
    """Main training function"""
    print("🎵 Hybrid Music Genre Classifier Training")
    print("=" * 50)
    
    # Paths
    CSV_PATH = r'C:\Users\Judah Edudzi\Downloads\archive\Data\features_30_sec.csv'
    AUDIO_FOLDER = r'C:\Users\Judah Edudzi\Downloads\archive\Data\features_30_sec.csv'  # Should contain genre subfolders with audio files
    MODEL_PATH = 'hybrid_music_classifier.pkl'
    
    # Check if CSV exists, if not create sample data
    if not os.path.exists(CSV_PATH):
        logger.warning(f"CSV file not found at {CSV_PATH}")
        logger.info("Creating sample data for demonstration...")
        
        try:
            sample_df = create_sample_data()
            sample_df.to_csv(CSV_PATH, index=False)
            logger.info(f"Sample data saved to {CSV_PATH}")
            
            print("\n📝 IMPORTANT: This is sample data!")
            print("   Replace 'music_features.csv' with your actual feature data")
            print("   Your CSV should have extracted features and genre labels")
        except Exception as e:
            logger.error(f"Failed to create sample data: {e}")
            return
    else:
        logger.info(f"Found existing CSV file: {CSV_PATH}")
        # Quick validation of existing CSV
        try:
            test_df = pd.read_csv(CSV_PATH)
            logger.info(f"Existing CSV shape: {test_df.shape}")
            logger.info(f"Existing CSV columns: {list(test_df.columns)}")
        except Exception as e:
            logger.error(f"Error reading existing CSV: {e}")
            return
    
    # Initialize classifier
    classifier = HybridMusicGenreClassifier()
    
    # Train CSV model
    try:
        logger.info(f"Training CSV model from {CSV_PATH}")
        csv_accuracy = classifier.train_csv_model(CSV_PATH)
        print(f"✅ CSV Model Accuracy: {csv_accuracy:.4f}")
    except Exception as e:
        logger.error(f"CSV training failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return
    
    # Train audio model (optional)
    if os.path.exists(AUDIO_FOLDER):
        try:
            logger.info(f"Training audio model from {AUDIO_FOLDER}")
            audio_accuracy = classifier.train_audio_model(AUDIO_FOLDER)
            print(f"✅ Audio Model Accuracy: {audio_accuracy:.4f}")
        except Exception as e:
            logger.warning(f"Audio training failed: {e}")
            logger.info("Continuing with CSV-only model...")
    else:
        logger.info(f"Audio folder not found: {AUDIO_FOLDER}")
        logger.info("Training with CSV data only")
    
    # Save the model
    try:
        classifier.save_model(MODEL_PATH)
        print(f"✅ Model saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return
    
    # Display available genres
    try:
        genres = list(classifier.label_encoder.classes_)
        print(f"\n🎭 Available genres: {genres}")
    except Exception as e:
        logger.warning(f"Could not display genres: {e}")
    
    print("\n🚀 Training completed successfully!")
    print(f"   Model file: {MODEL_PATH}")
    print("   You can now run the Flask app: python app.py")
    
if __name__ == "__main__":
    main()
