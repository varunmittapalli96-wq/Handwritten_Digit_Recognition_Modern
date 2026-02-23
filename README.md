# Handwritten Digit Recognition - Modern Implementation

A modern web-based handwritten digit recognition system built with Streamlit and TensorFlow, featuring an interactive drawing canvas and real-time prediction capabilities.

## 🚀 Features

- **Interactive Drawing Canvas**: Draw digits directly in your browser
- **Real-time Prediction**: Get instant predictions as you draw
- **Debug Mode**: View intermediate preprocessing steps
- **Image Processing Pipeline**: Advanced preprocessing with automatic inversion detection
- **Model Confidence**: See prediction confidence scores and top-3 predictions
- **Sample Saving**: Save drawn samples for further analysis

## 🛠️ Technology Stack

- **Frontend**: Streamlit with streamlit-drawable-canvas
- **Backend**: Python with TensorFlow/Keras
- **Model**: Trained on EMNIST dataset
- **Image Processing**: PIL, OpenCV, NumPy

## 📁 Project Structure

```
Handwritten_Digit_Recognition_Modern/
├── app/
│   └── app.py                 # Main Streamlit application
├── models/
│   └── emnist_digit_recognizer.h5  # Trained model
├── user_data/                 # Saved user drawings
├── venv/                      # Virtual environment
└── README.md                  # This file
```

## 🔧 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/varunmittapalli96-wq/Handwritten_Digit_Recognition_Modern.git
   cd Handwritten_Digit_Recognition_Modern
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit streamlit-drawable-canvas tensorflow pillow opencv-python numpy
   ```

## 🚀 Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app/app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Draw a digit** on the canvas and click "Predict / Debug" to see the results

## 🎯 How It Works

### Image Processing Pipeline

1. **Canvas Input**: User draws on a 250x250 black canvas with white strokes
2. **Compositing**: RGBA canvas data is composited onto black background
3. **Resizing**: Image is resized to 280x280 to preserve detail
4. **Auto-Inversion**: Automatically detects if inversion is needed based on brightness
5. **Thresholding**: Applies binary thresholding for clean digit extraction
6. **Bounding Box**: Finds the minimal bounding box around the digit
7. **Normalization**: Crops, scales to 20px max dimension, centers in 28x28 canvas
8. **Final Processing**: Applies Gaussian blur and normalizes to 0-1 range

### Model Architecture

The model is trained on the EMNIST dataset and expects:
- Input: 28x28 grayscale images
- Format: White digits on black background
- Normalization: Pixel values between 0-1

## 🎮 Usage Tips

- **Drawing**: Use thick strokes (stroke width: 20) for better recognition
- **Auto-Invert**: Enable for automatic brightness-based inversion
- **Manual Invert**: Force inversion if auto-detection fails
- **Debug Mode**: Check "Show all intermediate images" to see processing steps
- **Threshold Adjustment**: Use the slider to fine-tune thresholding

## 📊 Features Explained

### Debug Controls
- **Auto invert**: Automatically inverts based on brightness analysis
- **Force invert manually**: Manual override for inversion
- **Threshold slider**: Adjust binary thresholding value
- **Show all intermediate images**: Display preprocessing steps

### Prediction Output
- **Main prediction**: The digit with highest confidence
- **Confidence score**: Percentage confidence of the prediction
- **Top 3 predictions**: Shows alternative predictions with scores

## 🔍 Troubleshooting

**Low accuracy predictions?**
- Ensure digits are drawn clearly with thick strokes
- Try enabling/disabling auto-invert
- Check the debug images to see if preprocessing is correct

**Model not loading?**
- Verify the model file exists in `models/emnist_digit_recognizer.h5`
- Check that TensorFlow is properly installed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Varun Mittapalli**
- GitHub: [@varunmittapalli96-wq](https://github.com/varunmittapalli96-wq)

## 🙏 Acknowledgments

- EMNIST dataset for training data
- Streamlit team for the amazing framework
- TensorFlow team for the ML framework