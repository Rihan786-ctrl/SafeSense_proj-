# SafeSense

## Real-Time Human Activity Recognition & Risk Detection System

SafeSense is an AI-powered computer vision system designed to recognize
human activities from video streams and identify potentially risky
activities in real time.

The system combines computer vision, human pose/skeleton representation,
deep learning-based temporal activity recognition, risk assessment,
real-time inference and automated alert generation.

---

## Problem Statement

Traditional surveillance systems primarily depend on human monitoring
and predefined rule-based mechanisms. Continuous manual monitoring is
difficult to scale and may result in delayed detection of potentially
dangerous events.

SafeSense aims to provide an intelligent monitoring layer capable of
understanding human activities from video and identifying activities
that may require attention.

---

## Objectives

- Detect humans from video streams.
- Extract human pose/skeleton information.
- Recognize human activities using deep learning.
- Identify potentially risky activities.
- Assign risk levels to detected events.
- Generate real-time alerts.
- Provide a monitoring dashboard.
- Maintain an event and alert history.
- Provide measurable model and system performance.

---

## System Pipeline

Camera / Video
      ↓
Frame Processing
      ↓
Human Detection / Pose Estimation
      ↓
Skeleton & Motion Features
      ↓
Activity Recognition
      ↓
Risk Detection Engine
      ↓
Alert Generation
      ↓
Dashboard & Event Logging

---

## Technology Stack

### AI / Machine Learning
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn

### Computer Vision
- OpenCV
- Human Pose Estimation
- Skeleton-based representations

### Backend
- FastAPI
- Python

### Frontend
- To be finalized during implementation

### Database
- To be finalized based on deployment requirements

### MLOps
- Git
- GitHub
- DVC
- MLflow

### Deployment
- Docker
- Cloud deployment

---

## Repository Structure

```text
SafeSense/
├── backend/
├── configs/
├── data/
├── docs/
├── frontend/
├── models/
├── notebooks/
├── scripts/
├── src/
├── tests/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
