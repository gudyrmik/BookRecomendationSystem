from flask import Flask, render_template, request
from models import BookRecomendation

app = Flask(__name__)


@app.route('/')
def books():
    return render_template('books_index.html')


@app.route('/books_recommendation', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = BookRecomendation.recommend(request.form)
        return render_template("books_recommendation.html", result=result)


if __name__ == '__main__':
    app.run()
