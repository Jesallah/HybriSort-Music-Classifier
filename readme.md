# 🎵 Hybrid Music Genre Classifier

A sophisticated music genre classification system that combines pre-extracted features (70%) with real-time audio analysis (30%) for superior accuracy. Built for the "Hack with the Beat" hackathon.

## ✨ Features

- **Hybrid Classification**: Combines CSV pre-extracted features with real-time audio analysis
- **Beautiful UI**: Modern black and cyan interface with smooth animations
- **Spotify Integration**: Get personalized song recommendations based on detected genre
- **Multiple Audio Formats**: Supports MP3, WAV, FLAC, and M4A files
- **Real-time Analysis**: Fast genre prediction with confidence scores

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd music-genre-classifier

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Spotify API (Optional)

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications)
2. Create a new app
3. Copy your Client ID and Client Secret
4. Create a `.env` file:

```env
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

### 3. Train the Model

```bash
# Train with sample data (for testing)
python train_model.py

# Or train with your own data
# Place your CSV file as 'music_features.csv'
# Place audio files in 'audio_dataset/genre_name/' folders
python hybrid_model_trainer.py
```

### 4. Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` to use the application!

## 📁 Project Structure

```
music-genre-classifier/
├── app.py                      # Flask backend server
├── hybrid_model_trainer.py     # Hybrid model training logic
├── train_model.py              # Simple training script
├── templates/
│   └── index.html              # Beautiful UI template
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── music_features.csv          # Your CSV data (place here)
├── audio_dataset/              # Audio training data (optional)
│   ├── rock/
│   ├── pop/
│   ├── jazz/
│   └── ...
└── temp_uploads/               # Temporary file storage
```

## 🔬 How the Hybrid Model Works

The hybrid classifier combines two approaches:

1. **CSV Model (70% weight)**: Trained on pre-extracted features from your CSV file
2. **Audio Model (30% weight)**: Extracts features in real-time from uploaded audio

### Feature Extraction

The system extracts 33 audio features:
- Spectral features (centroid, rolloff, bandwidth)
- Zero-crossing rate
- 13 MFCC coefficients (mean & std)
- Chroma features
- Tempo
- RMS energy

## 📊 Training Your Own Model

### CSV Data Format

Your `music_features.csv` should have:
- Feature columns (33 features as extracted by librosa)
- Last column: `genre` (the target label)

Example:
```csv
spectral_centroid_mean,spectral_centroid_std,...,tempo,rms_mean,rms_std,genre
1854.23,823.45,...,120.5,0.15,0.08,rock
2341.67,945.12,...,128.3,0.18,0.09,pop
...
```

### Audio Data Structure

Organize audio files by genre:
```
audio_dataset/
├── rock/
│   ├── song1.mp3
│   ├── song2.wav
│   └── ...
├── pop/
│   ├── song1.mp3
│   └── ...
└── jazz/
    └── ...
```

## 🎨 UI Features

- **Drag & Drop**: Easy file uploading
- **Real-time Feedback**: Visual confidence scores
- **Spotify Integration**: Click tracks to open in Spotify
- **Responsive Design**: Works on desktop and mobile
- **Beautiful Animations**: Smooth transitions and effects

## 🔧 API Endpoints

- `GET /`: Main UI page
- `GET /api/health`: System health check
- `POST /api/classify`: Upload and classify audio file
- `GET /api/genres`: Get available genres

## 🎯 Customization

### Adjusting Hybrid Weights

In `hybrid_model_trainer.py`:
```python
self.csv_weight = 0.7    # 70% CSV model
self.audio_weight = 0.3  # 30% audio model
```

### Adding More Genres

Simply include them in your training data. The system automatically detects available genres.

### UI Customization

Edit the CSS variables in `templates/index.html`:
```css
:root {
    --cyan: #00ffff;        /* Primary accent color */
    --primary-bg: #0a0a0a;  /* Background color */
    /* ... */
}
```

## 🛠️ Troubleshooting

### Model Not Loading
- Ensure you've run `python train_model.py` first
- Check that `hybrid_music_classifier.pkl` exists

### Spotify Not Working
- Verify your `.env` file has correct credentials
- Check that your Spotify app has the right permissions

### Audio Processing Errors
- Ensure ffmpeg is installed (required by librosa)
- Check that your audio files are in supported formats

## 📈 Performance Tips

1. **Better Training Data**: More diverse, high-quality audio samples
2. **Feature Engineering**: Experiment with additional audio features
3. **Model Tuning**: Adjust hyperparameters in the training scripts
4. **Balanced Dataset**: Ensure equal representation across genres

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

This project is open source and available under the MIT License.

## 🏆 Hackathon Notes

Built for "Hack with the Beat" - showcasing the power of hybrid machine learning approaches in music technology!

---

**Happy Classifying! 🎵✨**