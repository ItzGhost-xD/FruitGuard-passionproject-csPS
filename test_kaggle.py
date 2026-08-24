import os
os.environ['KAGGLE_USERNAME'] = 'fruitguard_user'
os.environ['KAGGLE_KEY'] = 'dbc1bb8183a3baa0697816959433b039'

from kaggle import KaggleApi

api = KaggleApi()
api.authenticate()

print("Kaggle API authenticated successfully!")