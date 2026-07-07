#!/usr/bin/env python3
"""
Dual Hardware PWM Servo control (Pan/Tilt) on Raspberry Pi 5.

Wiring:
- Pan Servo (GPIO 12): Physical Pin 32
- Tilt Servo (GPIO 13): Physical Pin 33
"""

import sys
import tty
import termios
import select
import time
from rpi_hardware_pwm import HardwarePWM

MIN_ANGLE = -90
MAX_ANGLE = 90
STEP_DEG = 5

MIN_DUTY = 2.5
MAX_DUTY = 12.0

def angle_to_duty_cycle(angle):
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    return ((angle - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)) * (MAX_DUTY - MIN_DUTY) + MIN_DUTY

# Initialize both servos on chip 0
try:
    pan_servo = HardwarePWM(pwm_channel=0, hz=50, chip=0)
    tilt_servo = HardwarePWM(pwm_channel=1, hz=50, chip=0)
except Exception as e:
    print(f"Error initializing PWM: {e}")
    sys.exit(1)

pan_angle = 0
tilt_angle = 0

pan_servo.start(angle_to_duty_cycle(pan_angle))
tilt_servo.start(angle_to_duty_cycle(tilt_angle))

print('Dual Servo Control Started.')
print('Pan: A (left) / D (right)')
print('Tilt: W (up) / S (down)')
print('Quit: Q')

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)

try:
    tty.setcbreak(fd)
    while True:
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            ch = sys.stdin.read(1)
            if not ch: continue
            ch = ch.lower()
            
            if ch == 'q':
                print('\nQuitting...')
                break
            elif ch == 'a':
                pan_angle = min(MAX_ANGLE, pan_angle + STEP_DEG)
                pan_servo.change_duty_cycle(angle_to_duty_cycle(pan_angle))
                print(f'Pan moved left -> {pan_angle}°')
            elif ch == 'd':
                pan_angle = max(MIN_ANGLE, pan_angle - STEP_DEG)
                pan_servo.change_duty_cycle(angle_to_duty_cycle(pan_angle))
                print(f'Pan moved right -> {pan_angle}°')
            elif ch == 'w':
                tilt_angle = max(MIN_ANGLE, tilt_angle - STEP_DEG)
                tilt_servo.change_duty_cycle(angle_to_duty_cycle(tilt_angle))
                print(f'Tilt moved up -> {tilt_angle}°')
            elif ch == 's':
                tilt_angle = min(MAX_ANGLE, tilt_angle + STEP_DEG)
                tilt_servo.change_duty_cycle(angle_to_duty_cycle(tilt_angle))
                print(f'Tilt moved down -> {tilt_angle}°')
                
        time.sleep(0.05)
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    pan_servo.stop()
    tilt_servo.stop()
    print('Cleaned up hardware PWM and terminal.')