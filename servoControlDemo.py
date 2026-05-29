from gpiozero import AngularServo
from time import sleep

# Use a standard pulse range (1ms to 2ms) to start safely
# 0.001 is 1ms, 0.002 is 2ms
servo = AngularServo(14, min_angle=0, max_angle=180, 
                     min_pulse_width=0.001, max_pulse_width=0.002)

def set_angle(angle):
    if 0 <= angle <= 180:
        print(f"Moving to {angle}...")
        servo.angle = angle
        sleep(1) # Wait 1 second for the move to finish
    else:
        print("Angle out of range!")

try:
    while True:
        user_input = input("Enter angle 0 to 180 (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        angle = int(user_input)
        set_angle(angle)

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    servo.detach() # Stop sending signal to prevent jitter