def recommend(book_title):

    print(f"recommend called with {book_title} param")

    RATINGS_NUMBER_THRESHOLD = 8

    # import
    import pandas as pd
    import numpy as np

    # load ratings
    ratings = pd.read_csv('datasets\\BX-Book-Ratings.csv', encoding='cp1251', sep=';', warn_bad_lines=False)
    ratings = ratings[ratings['Book-Rating'] != 0]

    # load books
    books = pd.read_csv('datasets\\BX-Books.csv', encoding='cp1251', sep=';', error_bad_lines=False, warn_bad_lines=False)

    # users_ratigs = pd.merge(ratings, users, on=['User-ID'])
    dataset = pd.merge(ratings, books, on=['ISBN'])
    dataset_lowercase = dataset.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

    # get author of requested book
    author_arr = np.unique(dataset_lowercase.loc[dataset_lowercase['Book-Title'] == book_title]['Book-Author'])
    author = author_arr[1] if author_arr.any() else ''

    requested_author_users = dataset_lowercase['User-ID'][
        (dataset_lowercase['Book-Title'] == book_title) & (dataset_lowercase['Book-Author'] == author)]
    requested_author_users = np.unique(requested_author_users.tolist())

    # final dataset
    books_of_requested_author_users = dataset_lowercase[(dataset_lowercase['User-ID'].isin(requested_author_users))]

    # Number of ratings per other books in dataset
    number_of_rating_per_book = books_of_requested_author_users.groupby(['Book-Title']).agg('count').reset_index()

    # select only books which have actually higher number of ratings than threshold
    books_to_compare = number_of_rating_per_book['Book-Title'][
        number_of_rating_per_book['User-ID'] >= RATINGS_NUMBER_THRESHOLD]
    books_to_compare = books_to_compare.tolist()

    ratings_data_raw = books_of_requested_author_users[['User-ID', 'Book-Rating', 'Book-Title']][
        books_of_requested_author_users['Book-Title'].isin(books_to_compare)]

    # group by User and Book and compute mean
    ratings_data_raw_nodup = ratings_data_raw.groupby(['User-ID', 'Book-Title'])['Book-Rating'].mean()

    # reset index to see User-ID in every row
    ratings_data_raw_nodup = ratings_data_raw_nodup.to_frame().reset_index()

    dataset_for_corr = ratings_data_raw_nodup.pivot(index='User-ID', columns='Book-Title', values='Book-Rating')

    result_list = []
    worst_list = []

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
        tab = (ratings_data_raw[ratings_data_raw['Book-Title'] == other_book_title].groupby(
            ratings_data_raw['Book-Title']).mean())
        avgrating.append(tab['Book-Rating'].min())
        # final dataframe of all correlation of each book
    corr_fellowship = pd.DataFrame(list(zip(book_titles, correlations, avgrating)),
                                   columns=['book', 'corr', 'avg_rating'])
    corr_fellowship.head()

    # top 10 books with highest corr
    result_list.append(corr_fellowship.sort_values('corr', ascending=False).head(10))

    # worst 10 books
    worst_list.append(corr_fellowship.sort_values('corr', ascending=False).tail(10))

    # print("Correlation for book:", LoR_list[0])
    # print("Average rating of LOR:", ratings_data_raw[ratings_data_raw['Book-Title']=='the fellowship of the ring (the lord of the rings, part 1'].groupby(ratings_data_raw['Book-Title']).mean()))
    rslt = result_list[0]

    print(rslt)

    return rslt