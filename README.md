# Final-Project

PROJECT REPORT: MOVIE RECOMMENDERSYSTEM USING CONTENT-BASED FILTERING

1. INTRODUCTION
This project is a movie recommendation app built with Python and Streamlit. It helps users find similar movies based on the genre of a movie they like. Whena user selects a movie, the app suggests 5 others that share similar genres and arehighly rated by users. It also shows IMDb links so users can quickly check moredetails.

2. GOALS
 Recommend 5 similar movies based on genre.
 Sort suggestions by average user ratings.
 Make it easy and fun to use with a simple web interface.
 Show IMDb links and add some handpicked movie suggestions by genre.

3. HOW IT WORKS
 The app uses three datasets:
 movies.csv for titles and genres.
 ratings.csv for user ratings.
 links.csv for IMDb IDs.
 Genres are processed using TF-IDF (a text analysis technique).
 A Nearest Neighbors model finds movies with similar genre patterns.
 Average ratings are calculated and used to sort the results.
 IMDb links are generated using movie IDs, giving users direct access to moreinfo.

4. FEATURES
 Clean interface where users can:
 Select a movie from a dropdown.
 Click a button to get recommendations.
 See movie details like genres, rating, and IMDb link.
 A sidebar with handpicked movie picks in genres like Action, Comedy, Drama, and Sci-Fi.

5. RESULTS
The app provides quick, useful recommendations based on genre similarity andaverage ratings. It’s simple but effective, and great for users who want to exploremore movies like ones they already enjoyed.
