from flask import Flask, render_template, request
from models import BookRecomendation
from models.BookRecomendation import Recommender

app = Flask(__name__)
Recommender.load_data()

@app.route('/')
def books():
    return render_template('books_index.html')


@app.route('/books_recommendation', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = BookRecomendation.Recommender.recommend(request.form['Book title'])
        return render_template("books_recommendation.html", result=result)


if __name__ == '__main__':
    app.run()
