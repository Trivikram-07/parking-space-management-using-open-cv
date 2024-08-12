import cv2
import pymysql
import pytesseract
import time
import re
import threading
import numpy as np
from PIL import Image

# Function to insert a null row into the table
def insert_null_row(cursor, table_name):
    cursor.execute(f"INSERT INTO {table_name} (plate_number, capture_date) VALUES (NULL, NULL)")

# Image preprocessing function
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    edges = cv2.Canny(enhanced, 100, 200)
    return edges

# Improved plate recognition function
def recognize_plate(img):
    processed_img = preprocess_image(img)
    
    configs = [
        '--psm 8 --oem 3',
        '--psm 7 --oem 3',
        '--psm 6 --oem 3'
    ]
    
    results = []
    for config in configs:
        text = pytesseract.image_to_string(processed_img, config=config)
        cleaned_text = clean_text(text)
        if cleaned_text:
            results.append(cleaned_text)
    
    return most_common(results) if results else None

def clean_text(text):
    # Convert to uppercase
    text = text.upper()
    
    # Remove unwanted characters
    cleaned = re.sub(r'[^A-Z0-9]', '', text)
    
    # Remove 'P' from the start if present
    cleaned = re.sub(r'^P+', '', cleaned)
    
    # Add more specific patterns here based on your region's number plate format
    # For example, if your plates always have 2 letters followed by 4 numbers:
    # if re.match(r'^[A-Z]{2}\d{4}$', cleaned):
    #     return cleaned
    # else:
    #     return None
    
    return cleaned if cleaned else None

def most_common(lst):
    if not lst:
        return None
    return max(set(lst), key=lst.count)

# Function to process incoming vehicles
def process_incoming(parking_capacity, db_details, lock, available_spots, stop_event):
    connection = pymysql.connect(**db_details)
    cursor = connection.cursor()

    insert_null_row(cursor, "numberplate")

    plate_cascade = cv2.CascadeClassifier("model.xml")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream or file")
        return

    cap.set(3, 640)
    cap.set(4, 480)

    min_area = 500
    count = 0
    last_capture_time = time.time()

    while not stop_event.is_set():
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image")
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        current_time = time.time()
        if current_time - last_capture_time >= 12:
            for (x, y, w, h) in plates:
                area = w * h
                if area > min_area:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                    img_roi = img[y:y+h, x:x+w]
                    img_path = f"plates/plate_{count}.jpg"
                    cv2.imwrite(img_path, img_roi)
                    count += 1
                    
                    recognized_text = recognize_plate(img_roi)
                    
                    if recognized_text:
                        print("Recognized text:", recognized_text)

                        sql_insert_query = """INSERT INTO numberplate (plate_number, capture_date) VALUES (%s, NOW())"""
                        cursor.execute(sql_insert_query, (recognized_text,))
                        connection.commit()
                        print("License plate number stored in database:", recognized_text)

                        with lock:
                            if available_spots[0] > 0:
                                available_spots[0] -= 1
                            print(f"Available parking spots: {available_spots[0]}")

                            if available_spots[0] == 0:
                                print("Parking lot is full. Please come again later.")
                                break

            last_capture_time = current_time

        cv2.imshow("Incoming Vehicles", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or stop_event.is_set():
            break

    cap.release()
    cv2.destroyAllWindows()
    connection.close()

# Function to process outgoing vehicles
def process_outgoing(parking_capacity, db_details, lock, available_spots, stop_event):
    connection = pymysql.connect(**db_details)
    cursor = connection.cursor()

    insert_null_row(cursor, "numberplate_out")

    plate_cascade = cv2.CascadeClassifier("model.xml")

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video stream or file")
        return

    cap.set(3, 640)
    cap.set(4, 480)

    min_area = 500
    count = 0
    last_capture_time = time.time()

    while not stop_event.is_set():
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image")
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        current_time = time.time()
        if current_time - last_capture_time >= 12:
            for (x, y, w, h) in plates:
                area = w * h
                if area > min_area:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                    img_roi = img[y:y+h, x:x+w]
                    img_path = f"plates/plate_out_{count}.jpg"
                    cv2.imwrite(img_path, img_roi)
                    count += 1
                    
                    recognized_text = recognize_plate(img_roi)
                    
                    if recognized_text:
                        print("Recognized text:", recognized_text)

                        sql_insert_query = """INSERT INTO numberplate_out (plate_number, capture_date) VALUES (%s, NOW())"""
                        cursor.execute(sql_insert_query, (recognized_text,))
                        connection.commit()
                        print("License plate number stored in outgoing database:", recognized_text)

                        with lock:
                            if available_spots[0] < parking_capacity:
                                available_spots[0] += 1
                            print(f"Available parking spots: {available_spots[0]}")

            last_capture_time = current_time

        cv2.imshow("Outgoing Vehicles", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or stop_event.is_set():
            break

    cap.release()
    cv2.destroyAllWindows()
    connection.close()

# Function to monitor keyboard input for stopping the process
def monitor_keyboard(stop_event):
    input("Press Enter to stop...\n")
    stop_event.set()

# Main function
if __name__ == "__main__":
    db_details = {
        'host': 'localhost',
        'user': 'root',
        'password': 'root',
        'database': 'imagecollection'
    }

    parking_capacity = int(input("Enter the parking lot capacity: "))
    available_spots = [parking_capacity]
    lock = threading.Lock()
    stop_event = threading.Event()

    connection = pymysql.connect(**db_details)
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS numberplate (
            id INT AUTO_INCREMENT PRIMARY KEY,
            plate_number VARCHAR(255),
            capture_date DATETIME
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS numberplate_out (
            id INT AUTO_INCREMENT PRIMARY KEY,
            plate_number VARCHAR(255),
            capture_date DATETIME
        )
    """)
    connection.commit()
    connection.close()

    incoming_thread = threading.Thread(target=process_incoming, args=(parking_capacity, db_details, lock, available_spots, stop_event))
    outgoing_thread = threading.Thread(target=process_outgoing, args=(parking_capacity, db_details, lock, available_spots, stop_event))
    keyboard_thread = threading.Thread(target=monitor_keyboard, args=(stop_event,))

    incoming_thread.start()
    outgoing_thread.start()
    keyboard_thread.start()

    incoming_thread.join()
    outgoing_thread.join()
    keyboard_thread.join()