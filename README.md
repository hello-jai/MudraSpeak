# 🤟 ISL Recognition - Indian Sign Language Detection Using CNN

This project is a real-time Indian Sign Language (ISL) recognition system using a Convolutional Neural Network (CNN) and OpenCV. It processes video input from your webcam and predicts hand gestures corresponding to ISL characters or words.

## 🧠 Model

The model (`model.h5`) is trained on grayscale images of ISL hand signs resized to 64x64 pixels. It outputs the predicted label using softmax classification.

## 🗂 Dataset

This project expects a folder named `dataset/`, where each subfolder is named after a sign class (e.g., `A`, `B`, `Hello`) and contains training images for that gesture.

```
dataset/
├── A/
├── B/
├── Hello/
...
```

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.x
- pip

### 📦 Install Dependencies

```bash
pip install numpy opencv-python keras tensorflow
```

> You may need to install `h5py` as well: `pip install h5py`

### 🏁 Run the Project

```bash
python isl_recognition.py
```

Press `q` to quit the recognition window.

---

## 🔍 Project Structure

```
├── isl_recognition.py    # Main script for webcam-based recognition
├── model.h5              # Trained CNN model
├── dataset/              # Dataset used for training (not included)
└── README.md             # Project documentation
```

---

## 🔧 Future Improvements

- ✅ Add FPS counter and confidence scores.
- 🔲 Include gesture region detection for better accuracy.
- 🔲 Add training script and dataset download instructions.
- 🔲 Extend to full words or sentences using sequence models.

---

## 🤝 Contributing

Feel free to fork the repository, open issues, or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- Public ISL datasets used for training (not distributed with this repo)
