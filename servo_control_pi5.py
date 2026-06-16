#!/usr/bin/env python3
"""
Hardware PWM Servo control for a 9g micro servo on Raspberry Pi 5.

Wiring (physical pins):
- Pin 4: 5V (power)
- Pin 6: GND (ground)
- Pin 32: BCM GPIO12 (Hardware PWM0 Channel 0)
"""

import sys
import tty
import termios
import select
import time
from rpi_hardware_pwm import HardwarePWM

# Servo configuration
MIN_ANGLE = -90
MAX_ANGLE = 90
STEP_DEG = 5

# Map your specific pulse widths (0.0005s to 0.0024s) to Duty Cycle percentages
# At 50Hz (20ms period): 0.5ms = 2.5%, 2.4ms = 12.0%
MIN_DUTY = 2.5
MAX_DUTY = 12.0

def angle_to_duty_cycle(angle):
    """Converts a given angle to the corresponding duty cycle percentage."""
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    return ((angle - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)) * (MAX_DUTY - MIN_DUTY) + MIN_DUTY

# Initialize Hardware PWM on PWM0 (GPIO 12)
# Depending on your specific Pi OS Bookworm kernel, the RP1 PWM chip registers as 2 or 0.
try:
    servo = HardwarePWM(pwm_channel=0, hz=50, chip=0)
except Exception:
    print('Failed to initialize PWM on chip 2, trying chip 0...')
    servo = HardwarePWM(pwm_channel=0, hz=50, chip=2)

# Start centered
current_angle = 0
servo.start(angle_to_duty_cycle(current_angle))

print('Hardware PWM Servo started. Press a (left), d (right), q (quit).')

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
                current_angle = max(MIN_ANGLE, current_angle - STEP_DEG)
                servo.change_duty_cycle(angle_to_duty_cycle(current_angle))
                print(f'Moved left -> angle={current_angle}')
            elif ch == 'd':
                current_angle = min(MAX_ANGLE, current_angle + STEP_DEG)
                servo.change_duty_cycle(angle_to_duty_cycle(current_angle))
                print(f'Moved right -> angle={current_angle}')
                
        time.sleep(0.05)
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    servo.stop()
    print('Cleaned up hardware PWM and terminal.')