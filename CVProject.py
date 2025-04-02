import cv2
import numpy as np
import pigpio
from picamera2 import Picamera2

# Servo pin configuration
SERVO_PIN = 17
CENTER_ANGLE = 90
DEADZONE = 20

# Initialize pigpio for hardware PWM
pi = pigpio.pi()
pi.set_servo_pulsewidth(SERVO_PIN, 1500)  # Center position (1500 microseconds = 90 degrees)

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (640, 480)}))
picam2.start()

yaw_angle = CENTER_ANGLE  # Start at center position

while True:
    # Capture frame from camera
    frame = picam2.capture_array()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks and find contours
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                           cv2.inRange(hsv, lower_red2, upper_red2))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest red object in the frame
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # Get the center of the frame for comparison
        frame_center_x = frame.shape[1] // 2
        error_x = center_x - frame_center_x

        # Only move the servo if the object is outside the deadzone
        if abs(error_x) > DEADZONE:
            if error_x < 0 and yaw_angle > 0:  # Move left
                yaw_angle += 1
            elif error_x > 0 and yaw_angle < 180:  # Move right
                yaw_angle -= 1

            # Convert the yaw angle to pulse width (500 to 2500 microseconds)
            pulse_width = 500 + (yaw_angle * 2000 // 180)  # Scale 0-180 degrees to 500-2500 microseconds
            pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

        # Draw bounding box and center marker on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({error_x})", (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the processed frame
    cv2.imshow("Red Object Tracking", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.close()
pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Turn off servo
pi.stop()
