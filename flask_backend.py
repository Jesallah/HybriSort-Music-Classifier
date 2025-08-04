import os
import logging
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Import our hybrid model
from hybrid_model_trainer import HybridMusicGenreClassifier

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Spotify client
spotify = None
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    try:
        client_credentials_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        logger.info("Spotify client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Spotify client: {e}")
else:
    logger.warning("Spotify credentials not found")

# Load hybrid model
model = None
MODEL_PATH = 'hybrid_music_classifier.pkl'

try:
    model = HybridMusicGenreClassifier.load_model(MODEL_PATH)
    logger.info("Hybrid model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.info("Please train the model first using hybrid_model_trainer.py")

# Genre mapping for Spotify
GENRE_MAPPING = {
    'rock': ['rock', 'classic-rock', 'alternative-rock', 'indie-rock'],
    'pop': ['pop', 'dance-pop', 'electropop', 'indie-pop'],
    'jazz': ['jazz', 'smooth-jazz', 'contemporary-jazz', 'jazz-fusion'],
    'classical': ['classical', 'orchestral', 'chamber-music', 'symphony'],
    'electronic': ['electronic', 'house', 'techno', 'ambient', 'edm'],
    'hip-hop': ['hip-hop', 'rap', 'trap', 'old-school-hip-hop'],
    'country': ['country', 'country-rock', 'alt-country', 'bluegrass'],
    'blues': ['blues', 'chicago-blues', 'delta-blues', 'electric-blues'],
    'reggae': ['reggae', 'dub', 'dancehall', 'roots-reggae'],
    'folk': ['folk', 'indie-folk', 'contemporary-folk', 'americana'],
    'metal': ['metal', 'heavy-metal', 'death-metal', 'black-metal'],
    'disco': ['disco', 'funk', 'soul', 'r-n-b']
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_spotify_recommendations(genre, limit=8):
    """Get Spotify recommendations based on genre"""
    if not spotify:
        return []
    
    try:
        # Clean genre name for Spotify
        clean_genre = genre.lower().replace(' ', '-')
        genre_seeds = GENRE_MAPPING.get(clean_genre, [clean_genre])
        
        # Try each genre seed until one works
        for seed_genre in genre_seeds:
            try:
                recommendations = spotify.recommendations(
                    seed_genres=[seed_genre],
                    limit=limit,
                    market='US'
                )
                
                tracks = []
                for track in recommendations['tracks']:
                    # Get artist names
                    artists = [artist['name'] for artist in track['artists']]
                    
                    track_info = {
                        'name': track['name'],
                        'artists': artists,
                        'artist_string': ', '.join(artists),
                        'album': track['album']['name'],
                        'spotify_url': track['external_urls']['spotify'],
                        'preview_url': track['preview_url'],
                        'popularity': track['popularity'],
                        'image': track['album']['images'][0]['url'] if track['album']['images'] else None
                    }
                    tracks.append(track_info)
                
                if tracks:  # If we got recommendations, return them
                    return tracks
                    
            except Exception as e:
                logger.warning(f"Failed with genre seed '{seed_genre}': {e}")
                continue
        
        # Fallback: search for tracks with genre in query
        try:
            results = spotify.search(q=f'genre:{genre}', type='track', limit=limit)
            tracks = []
            for track in results['tracks']['items']:
                artists = [artist['name'] for artist in track['artists']]
                track_info = {
                    'name': track['name'],
                    'artists': artists,
                    'artist_string': ', '.join(artists),
                    'album': track['album']['name'],
                    'spotify_url': track['external_urls']['spotify'],
                    'preview_url': track['preview_url'],
                    'popularity': track['popularity'],
                    'image': track['album']['images'][0]['url'] if track['album']['images'] else None
                }
                tracks.append(track_info)
            return tracks
        except Exception as e2:
            logger.error(f"Spotify fallback search failed: {e2}")
            return []
            
    except Exception as e:
        logger.error(f"Spotify recommendations error: {e}")
        return []

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'spotify_available': spotify is not None,
        'available_genres': list(model.label_encoder.classes_) if model else []
    })

@app.route('/api/classify', methods=['POST'])
def classify_music():
    """Main classification endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Supported: mp3, wav, flac, m4a'
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            logger.info(f"Classifying {filename}")
            
            # Make hybrid prediction using only audio file
            # (In a real scenario, you might also have CSV features to pass)
            genre, confidence, probabilities = model.predict(audio_path=temp_path)
            
            logger.info(f"Prediction: {genre} with {confidence:.2f}% confidence")
            
            # Get Spotify recommendations
            recommendations = []
            if spotify:
                logger.info(f"Getting Spotify recommendations for: {genre}")
                recommendations = get_spotify_recommendations(genre, limit=8)
                logger.info(f"Found {len(recommendations)} recommendations")
            
            # Prepare response
            response = {
                'success': True,
                'genre': genre,
                'confidence': round(confidence, 2),
                'recommendations': recommendations,
                'model_type': 'hybrid',
                'filename': filename
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(e)}'
            }), 500
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info("Temporary file cleaned up")
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {cleanup_error}")
    
    except Exception as e:
        logger.error(f"Request handling error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Request failed: {str(e)}'
        }), 500

@app.route('/api/genres')
def get_available_genres():
    """Get list of available genres"""
    if model:
        return jsonify({
            'success': True,
            'genres': list(model.label_encoder.classes_)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 50MB'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("🎵 Music Genre Classifier API Starting...")
    print(f"Model loaded: {'✅' if model else '❌'}")
    print(f"Spotify available: {'✅' if spotify else '❌'}")
    
    if not model:
        print("⚠️  WARNING: Model not loaded!")
        print("   Run: python hybrid_model_trainer.py")
    
    if not spotify:
        print("⚠️  WARNING: Spotify not configured!")
        print("   Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file")
    
    print("\n🚀 Server ready at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
