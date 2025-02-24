# Market Research Analysis Application

This application allows users to upload PDF files, ask questions, and receive AI-generated insights based on the content of the uploaded documents. The application consists of a **React frontend** and a **FastAPI backend**.

---
## Screenshot: 
![Screenshot](https://github.com/user-attachments/assets/f42d1023-a582-43c4-9a1d-bebcb1145955)

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
   - [Backend Setup](#backend-setup)
   - [Frontend Setup](#frontend-setup)
4. [Running the Application](#running-the-application)
5. [Usage](#usage)
6. [File Structure](#file-structure)
7. [Technologies Used](#technologies-used)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

---

## Features

- **Upload PDF Files**: Users can upload multiple PDF files for analysis.
- **Ask Questions**: Users can enter a query related to the content of the uploaded PDFs.
- **AI-Generated Insights**: The application uses a language model to generate responses based on the PDF content.
- **Responsive Design**: The frontend is designed to work seamlessly on both desktop and mobile devices.

---

## Prerequisites

Before setting up the application, ensure you have the following installed:

- **Node.js** (for the frontend)
- **Python 3.8+** (for the backend)
- **pip** (Python package manager)
- **Git** (optional, for cloning the repository)

---

## Setup

### Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   
2. Create a virtual environment:
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the required Python packages:
  pip install -r requirements.txt

4. Start the FastAPI server:
  uvicorn main:app --reload
The backend will be running at http://localhost:8000.

### Frontend Setup
Navigate to the frontend directory:
cd frontend

1. Install the required Node.js packages:
  npm install
2. Start the React development server:
  npm start

The frontend will be running at http://localhost:3000.

3. Running the Application
Ensure both the backend and frontend servers are running.

4. Open your browser and navigate to http://localhost:3000.

5 Use the application to upload PDFs, ask questions, and view AI-generated insights.

## Usage
1. Enter a Query:

Type your question in the input field (e.g., "What is the main topic of the document?").

2. Upload PDF Files:

Click the "Choose File" buttons to upload two PDF files.

3. Submit:

Click the "Submit" button to send the query and files to the backend for processing.

4. View Response:

The AI-generated response will be displayed below the form.

### File Structure
Copy
├── frontend/
│   ├── public/              # Static assets
│   ├── src/                 # React components and logic
│   │   ├── App.js           # Main React component
│   │   ├── styles.css       # CSS styles
│   │   └── index.js         # Entry point
│   ├── package.json         # Node.js dependencies
│   └── README.md            # Frontend documentation
├── backend/
│   ├── main.py              # FastAPI backend logic
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Backend documentation
└── README.md                # Project overview

### Technologies Used
- Frontend
React: A JavaScript library for building user interfaces.

Axios: A promise-based HTTP client for making API requests.

CSS: For styling the application.

- Backend
FastAPI: A modern Python web framework for building APIs.

PyTorch: For running the language model.

Sentence Transformers: For text embedding and retrieval.

PDFPlumber: For extracting text from PDF files.

