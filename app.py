from logging import debug
import pandas as pd

from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity





df = pd.read_csv('hollywood.csv')

count = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
count_matr = count.fit_transform(df['new'])
cos_sim = cosine_similarity(count_matr, count_matr)


df = df.reset_index()
titles = df['title']
indices = pd.Series(df.index, index=df['title'])

#simple recommendation
def rcmd(title):
    title = title.lower()
    

    try:
        idx = indices[title]
        sim_scores = list(enumerate(cos_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        movies = df.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)
        qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        qualified['weighted_rating'] = qualified.apply(lambda x : (x['vote_count']/ (x['vote_count']+m) * x['vote_average']) + (m/ (m + x['vote_count']) * C), axis = 1)

        qualified = qualified.sort_values('weighted_rating', ascending=False).head(10)
        l = list(qualified.title)
        return l
       
    except: 
        return ("oops not in the database.")
   
        

        
   

app = Flask(__name__, template_folder='templates')

@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='n')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')
    
    
if __name__== "__main__":
    app.run(debug=True)