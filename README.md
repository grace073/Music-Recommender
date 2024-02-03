# Music Recommender System

## Overview

This project implements a Music Recommender System using Python, Streamlit, NLTK, and scikit-learn. The system analyzes a Spotify dataset, applies text processing techniques, and calculates cosine similarity to provide song recommendations based on user selections. Users can rate songs, provide feedback, and view personalized recommendations.

## Features

- **Data Preprocessing:**
  - Cleans and preprocesses Spotify dataset, including limiting the dataset to the first 50 rows.
  - Converts text data to lowercase, removes specific characters, and performs stemming using NLTK's PorterStemmer.

- **TF-IDF Vectorization:**
  - Utilizes scikit-learn's TfidfVectorizer to convert processed text data into a TF-IDF matrix.

- **Cosine Similarity:**
  - Computes cosine similarity between songs based on the TF-IDF matrix.

- **Recommendation Function:**
  - Provides song recommendations using a precomputed similarity matrix.

- **User Feedback and Ratings:**
  - Stores user ratings and feedback in dictionaries for data persistence.
  - Implements a user-friendly interface for rating songs and providing optional feedback.

- **Streamlit App:**
  - Utilizes Streamlit for building a web application with an interactive user interface.
  - Includes features like showing the DataFrame, song selection, rating sliders, and feedback text area.

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`

## Project Structure

- `app.py`: Main Streamlit application script.
- `user_ratings.pkl`: Pickle file for storing user ratings.
- `user_feedback.pkl`: Pickle file for storing user feedback.
- `requirements.txt`: List of project dependencies.

## Getting Started

1. Clone the repository: `git clone https://github.com/grace073/Music-Recommender.git`
2. Navigate to the project directory: `cd Music-Recommender`
3. Import the dataset `spotify_millsongdata.csv` from kaggle
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `streamlit run app.py`

