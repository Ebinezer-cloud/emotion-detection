import cv2
from deepface import DeepFace

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    try:
        # Analyze emotion
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        # Get emotion
        emotion = result[0]['dominant_emotion']

        # Display emotion
        cv2.putText(
            frame,
            emotion.upper(),
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        print(e)

    # Show webcam
    cv2.imshow("Emotion Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()