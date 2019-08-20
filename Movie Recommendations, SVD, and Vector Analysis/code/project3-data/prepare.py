import pandas 
from datetime import datetime
#from surprise import SVD
#from surprise import Reader 
#from surprise import Dataset
#from surprise import accuracy
#from surprise.model_selection import train_test_split


def prepare_movielens_data(data_path):

    # get user gender, index user ids from 0 to (#user - 1)
    users = pandas.read_csv(data_path + 'u.user', sep='|', header=None, names=['id', 'age', 'gender', 'occupation', 'zip-code'])
    gender = pandas.DataFrame(users['gender'].apply(lambda x: int(x == 'M'))) # convert F/M to 0/1
    user_id = dict(zip(users['id'], range(users.shape[0]))) # mapping user id to linear index

    # the zero-th column is the id, and the second column is the release date
    movies = pandas.read_csv(data_path + 'u.item', sep='|', encoding = 'latin-1', header=None, usecols=[0, 1, 2],
                             names=['item-id', 'title', 'release-year'])

    bad_movie_ids = list(movies['item-id'].loc[movies['release-year'].isnull()]) # get movie ids with a bad release date

    movies = movies[movies['release-year'].notnull()] # item 267 has a bad release year, remove this item
    release_year = pandas.DataFrame(movies['release-year'].apply(lambda x: datetime.strptime(x, '%d-%b-%Y').year))
    movie_id = dict(zip(movies['item-id'], range(movies.shape[0]))) # mapping movie id to linear index

    # get ratings, remove ratings of movies with bad release years.
    rating_triples = pandas.read_csv(data_path + 'u.data', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    rating_triples = rating_triples[['user', 'item', 'rating']] # drop the last column
    rating_triples = rating_triples[~ rating_triples['item'].isin(bad_movie_ids)] # drop movies with bad release years

    # map user and item ids to user indices
    rating_triples['user'] = rating_triples['user'].map(user_id)
    rating_triples['item'] = rating_triples['item'].map(movie_id)

    # the following set assertions guarantees that the user ids are in [0, #users), and item ids are in [0, #items)
    assert(rating_triples['item'].unique().min() == 0)
    assert(rating_triples['item'].unique().max() == movies.shape[0] - 1)
    assert(rating_triples['user'].unique().min() == 0)
    assert(rating_triples['user'].unique().max() == users.shape[0] - 1)
    assert(rating_triples['item'].unique().shape[0] == movies.shape[0])
    assert(rating_triples['user'].unique().shape[0] == users.shape[0])

    # training/test set split
    rating_triples = rating_triples.sample(frac=1, random_state=2018).reset_index(drop=True) # shuffle the data
    train_ratio = 0.9
    train_size = int(train_ratio * rating_triples.shape[0])

    trainset = rating_triples.loc[0:train_size]
    testset = rating_triples.loc[train_size + 1:]

    return trainset, testset, gender, release_year


if __name__ == "__main__":

    # prepare data
    print('Extracting data from the ml-100k dataset ...')

    trainset, testset, gender, release_year = prepare_movielens_data(data_path='../ml-100k/')

    trainset.to_csv('trainset.csv', index=False)
    testset.to_csv('testset.csv', index=False)
    gender.to_csv('gender.csv', index=False)
    release_year.to_csv('release-year.csv', index=False)
    
    print('Done')

