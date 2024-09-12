import pickle
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the DataFrame and precomputed similarity matrix
df = pd.read_csv(r'C:\Users\User\OneDrive\Desktop\projects\AI\MRS\spotify_millsongdata.csv')
df = df.head(50).drop('link', axis=1).reset_index(drop=True)
df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex=True)

stemmer = PorterStemmer()

def tokenization(txt): 
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

df['text'] = df['text'].apply(lambda x: tokenization(x))

tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidvector.fit_transform(df['text'])
similarity = cosine_similarity(matrix)

def recommendation(selected_song):
    selected_song = selected_song.strip().lower()
    idx = df[df['song'].str.strip().str.lower() == selected_song].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:11]:
        song_name = df.iloc[m_id[0]].song
        artist_name = df.iloc[m_id[0]].artist
        songs.append((song_name, artist_name))

    return songs

# Load the data or create empty dictionaries for user ratings and feedback
try:
    user_ratings = pickle.load(open('user_ratings.pkl', 'rb'))
    user_feedback = pickle.load(open('user_feedback.pkl', 'rb'))
except (FileNotFoundError, EOFError):
    user_ratings = {}
    user_feedback = {}

def save_user_feedback(song_id, rating, feedback):
    if song_id in user_ratings:
        user_ratings[song_id].append(rating)
    else:
        user_ratings[song_id] = [rating]

    if song_id in user_feedback:
        user_feedback[song_id].append(feedback)
    else:
        user_feedback[song_id] = [feedback]

    pickle.dump(user_ratings, open('user_ratings.pkl', 'wb'))
    pickle.dump(user_feedback, open('user_feedback.pkl', 'wb'))

# Streamlit App
st.title('Music Recommender System')

# Sidebar with the option to show the DataFrame
if st.sidebar.checkbox('Show DataFrame'):
    st.subheader('Sample of the DataFrame:')
    st.write(df.head(10))

# User input for selecting a song
selected_song = st.selectbox('Select a Song:', df['song'].values)
selected_song = selected_song.strip().lower()

# Star rating for the selected song
user_rating = st.slider('Rate the Song:', 1, 5)

# Text input for feedback
user_feedback_text = st.text_area('Provide Feedback (Optional):')

if st.button('Submit Rating and Feedback'):
    # Get recommendations for the selected song
    recommended_music_names = recommendation(selected_song)

    # Save user ratings and feedback
    save_user_feedback(selected_song, user_rating, user_feedback_text)

    # Display selected song details
    selected_song_details = df[df['song'] == selected_song]
    if not selected_song_details.empty:
        st.write(f"**Artist:** {selected_song_details['artist'].values[0]}")
        st.write(f"**Song Name:** {selected_song}")

    # Access and display user_feedback_text
    st.write(f"**User Feedback:** {user_feedback_text}")

    # Display recommended songs
    st.subheader('Recommended Songs:')
    for i, (song_name, artist_name) in enumerate(recommended_music_names):
        st.write(f"{i+1}. **Artist:** {artist_name}, **Song Name:** {song_name}")

# Button to view feedback for a specific song
if st.button('View Feedback for Selected Song'):
    if selected_song in user_feedback:
        st.subheader(f'Feedback for {selected_song}:')
        for feedback in user_feedback[selected_song]:
            st.write(feedback)
    else:
        st.write(f'No feedback available for {selected_song}')
