import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st
import time

# Streamlit UI setup
st.set_page_config(layout='wide')
st.title("Math Problem Solver using OpenCV and Google Gemini")
st.write("""
This project uses MediaPipe, OpenCV, and Google's Gemini AI to recognize and solve handwritten math equations.
**How to Use**:
1. Write a math equation using one finger.
2. Raise **four fingers** to send it to Gemini for solving.
3. Raise **all five fingers** to erase the canvas.
""")

# Streamlit layout
column1, column2 = st.columns([2, 1])
with column1:
    frameWindow = st.image([])
with column2:
    st.title("Answer")
    outputText = st.subheader("")

# Configure Google Gemini AI
genai.configure(api_key="your_api_key")  # Replace with actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Try 1 or -1 if 0 doesn't work
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height
time.sleep(2)  # Allow the camera to warm up

# Check if the camera opened successfully
if not cap.isOpened():
    st.error(" Error: Could not open webcam.")
    cap.release()
    exit()

# Initialize the HandDetector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    """Detects hands and returns finger states and landmark list."""
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand1 = hands[0]        
        lmList = hand1["lmList"]        
        fingers = detector.fingersUp(hand1)        
        return fingers, lmList    
    return None

def draw(info, previousPosition, canvas):
    """Draws on the canvas based on finger movements."""
    fingers, lmlist = info
    currentPosition = None    
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up (drawing mode)
        currentPosition = lmlist[8][0:2]
        if previousPosition is None:
            previousPosition = currentPosition
        cv2.line(canvas, currentPosition, previousPosition, (255, 0, 255), 10)
    elif fingers == [1, 1, 1, 1, 1]:  # All five fingers up (clear canvas)
        canvas = np.zeros_like(canvas)
    return currentPosition, canvas

def sendToGemini(model, canvas, fingers):
    """Sends the drawn equation to Google Gemini AI for solving."""
    if fingers == [1, 1, 1, 1, 0]:  # Four fingers up (send input)
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem:", pil_image])
        return response.text
    return ""

# Initialize variables
previousPosition = None
canvas = None
outputResult = ""

# Main Loop
while True:
    success, img = cap.read()
    if not success or img is None:
        st.warning("Warning: Failed to capture frame from camera.")
        continue  # Skip this iteration if frame capture fails
    
    img = cv2.flip(img, 1)  # Flip horizontally for better user experience
    if canvas is None:
        canvas = np.zeros_like(img)
    
    info = getHandInfo(img)
    if info:
        fingers, lmlist = info
        previousPosition, canvas = draw(info, previousPosition, canvas)
        outputResult = sendToGemini(model, canvas, fingers)
    
    # Merge webcam feed with the canvas
    combinedImage = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    
    # Convert to RGB for Streamlit
    combinedImage = cv2.cvtColor(combinedImage, cv2.COLOR_BGR2RGB)
    
    # Display in Streamlit
    frameWindow.image(combinedImage, channels="RGB")
    
    if outputResult:
        outputText.text(outputResult)

    # Exit on a button click (works in Streamlit)
    if st.button("Exit"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
