# link to alpha report video
https://postjceac-my.sharepoint.com/:v:/g/personal/nirbo_post_jce_ac_il/IQDJ-PZpt7qQRZRPQikN1gN5AX5vDyQBsAnC9I0toYYLZOo

the link shown in alpha report is corrupted

# Smart Baby Monitor 

## Overview
The Smart Baby Monitor is an intelligent monitoring system designed to help parents keep their babies safe and supervised in real time.  
The system combines video and audio streaming with machine learning–based event detection to identify critical situations such as crying or unusual movement, and instantly notify parents through a mobile or web application.

This project is as a final-year software engineering project and focuses on real-time video and audio processing, alert system and mobile app.

---

## Key Features
- **Real-time video streaming**
- **Audio monitoring with cry detection**
- **Machine learning–based event detection**
- **Instant alerts and notifications**
- **Client–Server architecture**

---

## System Architecture
The system is built using a **Client–Server architecture**:

- **Server (Raspberry Pi 5)**  
  - Captures video and audio input  
  - Processes data in real time  
  - Runs ML models for event detection  
  - Manages alerts and logs  

- **Client (Mobile / Web App)**  
  - Displays live video and audio  
  - Receives notifications and alerts  
  - Allows remote monitoring  
  
---

## Technologies Used
- **Programming Languages:** Python, JavaScript  
- **Backend:** Python-based server  
- **Frontend:** Web / Mobile client (Android studio
- **Machine Learning:** Cry detection and audio classification models  
- **Hardware:** Raspberry Pi, Camera Module, Microphone  
- **Communication:** Client–Server communication over network  
- **Optional APIs:** Location / Notification services  

---

## Machine Learning Capabilities
The system integrates machine learning models to:
- Detect body posture
- Detect baby crying  
- Classify baby crying  


/home/nir/Desktop/server/web_server.py

to manualy stop or start the auto started web_server 
bash:
sudo systemctl stop webserver.service

sudo systemctl start webserver.service
