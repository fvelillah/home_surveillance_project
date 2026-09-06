# Project: Home security surveillance analytics

- Objective: Build a video analytics app for my home using our CCTV system
    - Real-time alerts with natural language descriptions of anomalous situations (robbery, vandalism, unusual behavior, etc)
    - Video search capabilities: example asking for latest incidents
    - Edge triage: using local embedding models to effectively escalate anomalous video clips
    - Cloud anomaly detection and classification: using high accuracy embedding models to detect anomalies and classifying them. Also using a VLM to describe in natural language the scene
    - kNN lookup: Anomaly score based on the cosine similarity with respect to the nearest neighbors in the vector space
    - UI with live camera, real-time alerts and video search
- CCTV setup: 
    - Dahua NVR DH-NVR5216-16P-4K
    - 9 IP cameras connected directly to the NVR (POW)
    
    