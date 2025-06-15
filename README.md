# 🚗 Parking Space Management System with ANPR

Welcome to the **Parking Space Management System**! This project leverages Automatic Number Plate Recognition (ANPR) to track vehicles entering and exiting a parking lot, managing available parking slots in real-time. Using two cameras (phone and PC/IP camera) connected via Wi-Fi, it detects number plates, updates slot counts, and displays free spaces to users. Built with Python, OpenCV, and Tesseract OCR, it enhances parking efficiency and automation.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" height="35" alt="python badge" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white&style=for-the-badge" height="35" alt="opencv badge" />
  <img src="https://img.shields.io/badge/Tesseract-000000?logo=google&logoColor=white&style=for-the-badge" height="35" alt="tesseract badge" />
  <img src="https://img.shields.io/github/stars/Trivikram-07/parking-space-management-using-open-cv?color=blueviolet&style=for-the-badge" height="35" alt="stars badge" />
</div>

---

## 📋 Overview

This system extends the ANPR capabilities to manage parking lots with the following features:
- **Entry Camera**: Detects number plates of incoming vehicles and reduces available slots.
- **Exit Camera**: Detects number plates of outgoing vehicles and increases available slots.
- **Slot Management**: Matches number plates to track vehicle status and updates free slot counts in real-time.
- **Display**: Shows available parking slots via a simple GUI.

---

## ✨ Features

- 🚘 **Real-Time ANPR**: Detects vehicle number plates using Haar Cascade Classifier and extracts text with Tesseract OCR.
- 📊 **Slot Tracking**: Decrements slots for incoming vehicles and increments for outgoing ones by matching number plates.
- 📸 **Dual Camera Support**: Integrates phone and IP/PC camera feeds via Wi-Fi for entry and exit monitoring.
- 🖥️ **User Interface**: Displays free parking slots using a Tkinter-based GUI.
- 💾 **Data Storage**: Logs number plates and timestamps in an SQLite database for tracking.
- 🔧 **Scalable**: Easily configurable for different parking lot sizes and camera setups.

---

## 🛠️ Setup Instructions

### Prerequisites
- **Hardware**:
  - PC with Windows/Linux/Mac.
  - Android/iOS phone with a camera.
  - IP camera (Wi-Fi or Ethernet, e.g., Reolink, Hikvision).
  - Wi-Fi router.
- **Software**:
  - Python 3.8+.
  - Tesseract OCR installed ([Tesseract Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)).
  - IP Webcam app (Android) or Camo app (iOS/Android).
  - IP camera with RTSP/HTTP streaming support.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Trivikram-07/parking-space-management-using-open-cv.git
   cd parking-space-management-using-open-cv
Install Dependencies:

```
Copy
Edit
pip install opencv-python numpy pytesseract tkinter sqlite3
```
Install Tesseract OCR:

Windows: Download from Tesseract GitHub and add it to PATH.

Linux: Run sudo apt-get install tesseract-ocr.

Mac: Install via Homebrew using brew install tesseract.

Configure Camera URLs:
Update the following variables in main.py:

python
```
Copy
Edit
ENTRY_CAMERA_URL = "http://192.168.1.101:8080/video"  # Example: IP webcam
EXIT_CAMERA_URL = "rtsp://admin:password@192.168.1.100:554/Streaming/channels/101"  # Example: IP camera
TOTAL_SLOTS = 50  # Total parking slots
Set Tesseract Path:
Ensure the Tesseract path is correctly set in main.py:
```
python
```
Copy
Edit
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update for your OS
Run the System:
```
bash
Copy
Edit
```
python main.py
```
# 🚀 How It Works

Incoming Vehicles:

Detected by the entry camera.

The number plate is recognized, and the count of available slots decreases.

Outgoing Vehicles:

Detected by the exit camera.

The system verifies the number plate and increases the count of available slots.

Real-Time Display:

A GUI updates in real-time to show available parking slots.

Database Logging:

Number plates and timestamps are logged into an SQLite database.

# 📈 Impact
Automation: Eliminates manual parking slot tracking, saving time and reducing errors.

Real-Time Updates: Provides instant slot availability, improving user experience.

Reliability: Accurate number plate detection enhances parking management efficiency.

Scalability: Suitable for small lots to large commercial parking facilities.

# 🤝 Contributing
Fork the repository.

Create a feature branch:

bash
Copy
Edit
```
git checkout -b feature/new-feature
```
Commit your changes:

bash
Copy
Edit
```
git commit -m "Add new feature"
```
Push to the branch:

bash
Copy
Edit
```
git push origin feature/new-feature
```
Open a Pull Request.
