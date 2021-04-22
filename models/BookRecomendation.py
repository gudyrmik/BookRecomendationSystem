import pandas as pd
import numpy as np
from datetime import datetime


class Recommender:
    """
    This constant represents a minimal number of any non-zero rates,
    that a book should have to be considered by a recommendation engine.
    """
    RATINGS_NUMBER_THRESHOLD = 8

    @staticmethod
    def load_data():
        """
        Reads RAW data from .csv files in a 'datasets' directory and prepares a dataset for recommendation engine usage in a
        Recommender.dataset field
        """
        # load ratings
        ratings_data = pd.read_csv('datasets\\BX-Book-Ratings.csv', encoding='cp1251', sep=';', low_memory=False,
                                   warn_bad_lines=False)
        ratings_data = ratings_data[ratings_data['Book-Rating'] != 0]

        # load books
        books_data = pd.read_csv('datasets\\BX-Books.csv', encoding='cp1251', sep=';', low_memory=False,
                                 error_bad_lines=False,
                                 warn_bad_lines=False)

        dataset = pd.merge(ratings_data, books_data, on=['ISBN'])
        Recommender.dataset = Recommender.data_preprocess(dataset)

    @staticmethod
    def data_preprocess(dataset):
        """
        Cleans RAW dataset from unused columns, ensures correct data types, fills gaps with meaningful strings.
        :param dataset: RAW dataset merged from source data files
        :return: dataset prepared for engine usage
        """
        # drop non-related fields
        dataset = dataset.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=False)
        dataset['Year-Of-Publication'] = pd.to_numeric(dataset['Year-Of-Publication'], 'coerce').fillna(
            datetime.now().year)
        dataset[['Book-Author', 'Publisher']] = dataset[['Book-Author', 'Publisher']].fillna(
            'NO DATA')

        dataset = dataset.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

        return dataset

    @staticmethod
    def create_dataset_for_correlations(book_title):
        """
        Takes RAW dataset, loaded into static Recommender.dataset_lowercase, and prepares it for correlation calculation
        considering user requested book title
        :param book_title: title of the book obtained from a user
        :return: dataset prepared for correlation calculation
        """

        # get author of requested book
        author_arr = np.unique(
            Recommender.dataset.loc[Recommender.dataset['Book-Title'] == book_title]['Book-Author'])
        author = author_arr[1] if author_arr.any() else ''

        requested_author_users = Recommender.dataset['User-ID'][
            (Recommender.dataset['Book-Title'] == book_title) & (
                    Recommender.dataset['Book-Author'] == author)]
        requested_author_users = np.unique(requested_author_users.tolist())

        # final dataset
        books_of_requested_author_users = Recommender.dataset[
            (Recommender.dataset['User-ID'].isin(requested_author_users))]

        # Number of ratings per other books in dataset
        number_of_rating_per_book = books_of_requested_author_users.groupby(['Book-Title']).agg('count').reset_index()

        # select only books which have actually higher number of ratings than threshold
        books_to_compare = number_of_rating_per_book['Book-Title'][
            number_of_rating_per_book['User-ID'] >= Recommender.RATINGS_NUMBER_THRESHOLD]
        books_to_compare = books_to_compare.tolist()

        Recommender.ratings_data_raw = books_of_requested_author_users[['User-ID', 'Book-Rating', 'Book-Title']][
            books_of_requested_author_users['Book-Title'].isin(books_to_compare)]

        # group by User and Book and compute mean
        ratings_data_raw_nodup = Recommender.ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()

        # reset index to see User-ID in every row
        ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()

        dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')
        return dataset_for_corr

    @staticmethod
    def calculate_correlation(book_title, dataset_for_corr):
        """
        Calculate correlations between books of the same author as user provided book, and others.
        :param book_title: title of the book obtained from a user
        :param dataset_for_corr: dataset used to calculate correlations
        :return: dataframe contained correlation books information
        """

        # Take out the selected book from correlation dataframe
        dataset_of_other_books = dataset_for_corr.copy(deep=False)
        if book_title in dataset_of_other_books:
            dataset_of_other_books.drop(book_title, axis=1, inplace=True)

        # empty lists
        book_titles = []
        correlations = []
        avgrating = []

        # corr computation
        for other_book_title in list(dataset_of_other_books.columns.values):
            book_titles.append(other_book_title)
            correlations.append(dataset_for_corr[book_title].corr(dataset_of_other_books[other_book_title]))
            tab = (Recommender.ratings_data_raw[Recommender.ratings_data_raw['Book-Title'] == other_book_title].groupby(
                Recommender.ratings_data_raw['Book-Title']).mean())
            avgrating.append(tab['Book-Rating'].min())
            # final dataframe of all correlation of each book
        corr_fellowship = pd.DataFrame(list(zip(book_titles, correlations, avgrating)),
                                       columns=['book', 'corr', 'avg_rating'])

        return corr_fellowship

    @staticmethod
    def recommend(book_title, number_of_results=10):
        """
        Primary API endpoint of books recommendation engine.
        :param book_title: title of the book obtained from a user
        :param number_of_results: number of books to recommend
        :return: list of recommended book titles
        """
        dataset_for_corr = Recommender.create_dataset_for_correlations(book_title)

        result_list = []
        worst_list = []

        dataset_with_correlations = Recommender.calculate_correlation(book_title, dataset_for_corr)

        # top 10 books with highest corr
        result_list.append(dataset_with_correlations.sort_values('corr', ascending=False).head(number_of_results))

        # worst 10 books
        worst_list.append(dataset_with_correlations.sort_values('corr', ascending=False).tail(number_of_results))

        # print("Correlation for book:", LoR_list[0])
        # print("Average rating of LOR:", ratings_data_raw[ratings_data_raw['Book-Title']=='the fellowship of the ring (the lord of the rings, part 1'].groupby(ratings_data_raw['Book-Title']).mean()))
        rslt = result_list[0]

        return rslt
