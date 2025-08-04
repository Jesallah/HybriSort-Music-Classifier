import pandas as pd
import numpy as np
import librosa
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridMusicGenreClassifier:
    """
    A hybrid classifier that combines pre-extracted features (70% weight) 
    with real-time audio feature extraction (30% weight)
    """
    
    def __init__(self):
        self.csv_model = None
        self.audio_model = None
        self.csv_scaler = StandardScaler()
        self.audio_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.csv_weight = 0.7
        self.audio_weight = 0.3
        
    def extract_audio_features(self, audio_path, duration=30):
        """Extract features from audio file - matches your existing extraction"""
        try:
            y, sr = librosa.load(audio_path, duration=duration)
            features = []
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([np.mean(chroma), np.std(chroma)])
            
            # Tempo
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            features.append(tempo)
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            return np.array(features)
        
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise
    
    def train_csv_model(self, csv_path):
        """Train model on pre-extracted CSV features"""
        logger.info("Training CSV-based model...")
        
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Check for missing values
            if df.isnull().any().any():
                logger.warning("Found missing values in CSV, filling with 0")
                df = df.fillna(0)
            
            # Identify non-numeric columns to exclude from features
            non_feature_columns = ['filename', 'genre', 'label']  # Common non-feature columns
            
            # Find the genre/label column (usually the last column or named 'genre'/'label')
            if 'genre' in df.columns:
                target_column = 'genre'
            elif 'label' in df.columns:
                target_column = 'label'
            else:
                # Assume last column is the target
                target_column = df.columns[-1]
                logger.info(f"No 'genre' or 'label' column found, using last column: {target_column}")
            
            # Get feature columns (exclude non-numeric and target columns)
            feature_columns = [col for col in df.columns 
                             if col not in non_feature_columns and col != target_column]
            
            logger.info(f"Target column: {target_column}")
            logger.info(f"Feature columns ({len(feature_columns)}): {feature_columns[:5]}...")  # Show first 5
            
            # Separate features and labels
            X_csv = df[feature_columns]
            y = df[target_column]
            
            logger.info(f"Features shape: {X_csv.shape}")
            logger.info(f"Labels shape: {y.shape}")
            logger.info(f"Unique genres: {y.unique()}")
            
            # Check if all feature columns are numeric
            non_numeric_cols = []
            for col in feature_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                logger.warning(f"Found non-numeric feature columns: {non_numeric_cols}")
                # Try to convert to numeric, replacing errors with NaN
                for col in non_numeric_cols:
                    X_csv[col] = pd.to_numeric(X_csv[col], errors='coerce')
                
                # Fill any resulting NaN values with 0
                X_csv = X_csv.fillna(0)
            
            # Convert to numpy arrays and ensure proper data types
            X_csv = X_csv.astype(np.float64).values
            y = y.astype(str).values
            
            # Check for any remaining non-numeric values
            if not np.isfinite(X_csv).all():
                logger.warning("Non-finite values found in features, cleaning...")
                # Replace inf and -inf with large finite numbers
                X_csv = np.nan_to_num(X_csv, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            logger.info(f"Encoded labels shape: {y_encoded.shape}")
            
            # Check if we have enough samples per class
            unique, counts = np.unique(y_encoded, return_counts=True)
            logger.info(f"Class distribution: {dict(zip(self.label_encoder.classes_[unique], counts))}")
            
            min_samples = min(counts)
            if min_samples < 2:
                logger.error("Not enough samples per class for train/test split!")
                raise ValueError("Need at least 2 samples per class")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_csv, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            logger.info(f"Training set shape: {X_train.shape}")
            logger.info(f"Test set shape: {X_test.shape}")
            
            # Scale features
            logger.info("Scaling features...")
            X_train_scaled = self.csv_scaler.fit_transform(X_train)
            X_test_scaled = self.csv_scaler.transform(X_test)
            
            # Train SVM model
            logger.info("Training SVM model...")
            self.csv_model = SVC(kernel='rbf', probability=True, random_state=42)
            self.csv_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.csv_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"CSV Model Accuracy: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error in train_csv_model: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def train_audio_model(self, audio_folder_path):
        """Train model on audio files organized in genre folders"""
        logger.info("Training audio-based model...")
        
        X_audio = []
        y_audio = []
        
        # Extract features from audio files
        for genre in os.listdir(audio_folder_path):
            genre_path = os.path.join(audio_folder_path, genre)
            if not os.path.isdir(genre_path):
                continue
                
            logger.info(f"Processing genre: {genre}")
            
            for filename in os.listdir(genre_path):
                if filename.endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    try:
                        audio_path = os.path.join(genre_path, filename)
                        features = self.extract_audio_features(audio_path)
                        X_audio.append(features)
                        y_audio.append(genre)
                    except Exception as e:
                        logger.warning(f"Failed to process {filename}: {e}")
        
        if not X_audio:
            logger.warning("No audio features extracted. Using dummy audio model.")
            self.audio_model = None
            return 0.0
        
        X_audio = np.array(X_audio)
        y_audio = np.array(y_audio)
        
        # Encode labels (reuse the same encoder)
        y_audio_encoded = self.label_encoder.transform(y_audio)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_audio, y_audio_encoded, test_size=0.2, random_state=42, stratify=y_audio_encoded
        )
        
        # Scale features
        X_train_scaled = self.audio_scaler.fit_transform(X_train)
        X_test_scaled = self.audio_scaler.transform(X_test)
        
        # Train Random Forest model (often better for audio features)
        self.audio_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.audio_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.audio_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Audio Model Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict(self, csv_features=None, audio_path=None):
        """
        Make hybrid prediction using both CSV features and audio file
        """
        if csv_features is None and audio_path is None:
            raise ValueError("Either csv_features or audio_path must be provided")
        
        predictions = []
        weights = []
        
        # CSV-based prediction
        if csv_features is not None and self.csv_model is not None:
            csv_features_scaled = self.csv_scaler.transform(csv_features.reshape(1, -1))
            csv_proba = self.csv_model.predict_proba(csv_features_scaled)[0]
            predictions.append(csv_proba)
            weights.append(self.csv_weight)
        
        # Audio-based prediction
        if audio_path is not None and self.audio_model is not None:
            audio_features = self.extract_audio_features(audio_path)
            audio_features_scaled = self.audio_scaler.transform(audio_features.reshape(1, -1))
            audio_proba = self.audio_model.predict_proba(audio_features_scaled)[0]
            predictions.append(audio_proba)
            weights.append(self.audio_weight)
        
        # If only one model is available, use it with full weight
        if len(predictions) == 1:
            final_proba = predictions[0]
        else:
            # Weighted average of predictions
            final_proba = np.average(predictions, axis=0, weights=weights)
        
        # Get prediction and confidence
        predicted_class = np.argmax(final_proba)
        confidence = np.max(final_proba) * 100
        genre = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return genre, confidence, final_proba
    
    def save_model(self, filepath):
        """Save the trained hybrid model"""
        model_data = {
            'csv_model': self.csv_model,
            'audio_model': self.audio_model,
            'csv_scaler': self.csv_scaler,
            'audio_scaler': self.audio_scaler,
            'label_encoder': self.label_encoder,
            'csv_weight': self.csv_weight,
            'audio_weight': self.audio_weight
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained hybrid model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls()
        classifier.csv_model = model_data['csv_model']
        classifier.audio_model = model_data['audio_model']
        classifier.csv_scaler = model_data['csv_scaler']
        classifier.audio_scaler = model_data['audio_scaler']
        classifier.label_encoder = model_data['label_encoder']
        classifier.csv_weight = model_data['csv_weight']
        classifier.audio_weight = model_data['audio_weight']
        
        return classifier

def main():
    """Train and save the hybrid model"""
    # Initialize classifier
    classifier = HybridMusicGenreClassifier()
    
    # Paths (update these to your actual paths)
    CSV_PATH = r'C:\Users\Judah Edudzi\Downloads\archive\Data\features_30_sec.csv'  # Your CSV with pre-extracted features
    AUDIO_FOLDER_PATH = r'C:\Users\Judah Edudzi\Downloads\archive\Data\genres_original'  # Folder with genre subfolders containing audio files
    MODEL_SAVE_PATH = 'hybrid_music_classifier.pkl'
    
    # Train CSV model
    if os.path.exists(CSV_PATH):
        csv_accuracy = classifier.train_csv_model(CSV_PATH)
        logger.info(f"CSV model trained with accuracy: {csv_accuracy:.4f}")
    else:
        logger.warning(f"CSV file not found: {CSV_PATH}")
    
    # Train audio model
    if os.path.exists(AUDIO_FOLDER_PATH):
        audio_accuracy = classifier.train_audio_model(AUDIO_FOLDER_PATH)
        logger.info(f"Audio model trained with accuracy: {audio_accuracy:.4f}")
    else:
        logger.warning(f"Audio folder not found: {AUDIO_FOLDER_PATH}")
    
    # Save the model
    classifier.save_model(MODEL_SAVE_PATH)
    logger.info("Hybrid model training completed!")
    
    # Test the model (optional)
    logger.info("Available genres:", classifier.label_encoder.classes_)

if __name__ == "__main__":
    main()