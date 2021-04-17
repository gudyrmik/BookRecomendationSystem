from flask import Flask

app = Flask(__name__)


@app.route('/')
def books_recomendation():
    render_template('books.html')


if __name__ == '__main__':
    app.run()
