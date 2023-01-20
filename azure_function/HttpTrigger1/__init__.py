
import logging
import azure.functions as func
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import azure.functions as func
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import time
from time import time
from surprise.model_selection import train_test_split
import numpy as np
from surprise import KNNWithMeans
import json

embeddings = pd.read_pickle('small_embeddings.pickle')
clicks = pd.read_csv("small_clicks.csv")
articles = pd.read_csv("articles_metadata.csv")

# Le publisher n'est pas une feature intéressante pour l'instant
articles.drop(columns=['publisher_id'], inplace=True)

# Conversion du ty'pe de données des metadat en nombres en entiers
articles = articles.astype(np.int64)


def predict_best_category_for_user(user_id, model, n_reco=5):
    '''Return n_reco recommended articles ID to user'''
    start = time()

    # Print l'Id de l'utilisateur
    print('The user id is:', user_id)
    
    predictions = {}
    
    for i in range(1, 460):
        _, category_id, _, est, err = model.predict(user_id, i)
        

        if (err != True):
            predictions[category_id] = est
    
    best_categories_to_recommend = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_reco])
    
    recommended_articles = []
    for key, _ in best_categories_to_recommend.items():
        recommended_articles.append(int(articles[articles['category_id'] == key]['article_id'].sample(1).values))
        
    print(f'Model executed in {round(time() - start, 2)}s')
    print(f'The recommended categories are: {list(best_categories_to_recommend.keys())}')
    print(f'1 random article from each categories: {recommended_articles}')
    
    return recommended_articles
    #, best_categories_to_recommend



sim_options = {
    "name": "cosine",
    # Compute similarities between items
    "user_based": False,  
}


dict_article_categories = articles.set_index('article_id')['category_id'].to_dict()


clicks['category_id'] = clicks['article_id'].map(dict_article_categories).astype(int)
clicks['total_click'] = clicks.groupby(['user_id'])['article_id'].transform('count')
clicks['total_click_by_category_id'] = clicks.groupby(['user_id','category_id'])['article_id'].transform('count')
clicks['rating'] = clicks['total_click_by_category_id'] / clicks['total_click']


reader = Reader(rating_scale=(0, 1))

data = Dataset.load_from_df(clicks[['user_id', 'category_id', 'rating']], reader)
train_set, test_set = train_test_split(data, test_size=.25)
model = KNNWithMeans(sim_options=sim_options).fit(train_set)

#user_id = 0
#n_reco = 5
#results, recommended_categories = predict_best_category_for_user(user_id, model, n_reco)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.params.get('user_id')
    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('user_id')

    if user_id:
        recommendation= predict_best_category_for_user(user_id, model, n_reco=5)
        # Convert the list in string
        str_result = ' '.join(str(elem) + "," for elem in recommendation)

        # Delete the last comma
        results = str_result.rstrip(str_result[-1])

        # Template example is to return a sentence with the user_id
        return func.HttpResponse(json.dumps(recommendation))
    
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
