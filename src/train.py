from model_utils import update_model, save_simple_metrics_report, get_model_performance_test_set
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PowerTransformer

import numpy as np
import sys
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

logger.info('Loading data...')
data = pd.read_csv('dataset/full_data.csv')

full_movie_data_scaled = data.copy()
scaler = PowerTransformer(method='box-cox')
full_movie_data_scaled = scaler.fit_transform(full_movie_data_scaled)
full_movie_data_scaled = pd.DataFrame(full_movie_data_scaled, columns = data.columns)

logger.info('Loading model...')
# mejoras 2.0
test_size = 0.20
imputer = KNNImputer(missing_values=np.nan)
cv = ShuffleSplit(n_splits=10, test_size=test_size, random_state=42)
n_jobs = -1
model = Pipeline([
    ('imputer', imputer),
    ('core_model', GradientBoostingRegressor())
])

logger.info('Splitting data...')
X = full_movie_data_scaled.drop(['worldwide_gross'], axis = 1)
y = full_movie_data_scaled['worldwide_gross']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

logger.info('Setting Hyperparameters...')
param_tunning = {'core_model__n_estimators': [245],
                'core_model__criterion': ['friedman_mse'],
                'core_model__learning_rate': [0.017],
                'core_model__loss': ['huber'],
                'core_model__max_depth': [2],
                'core_model__min_samples_leaf': [3],
                'core_model__min_samples_split': [3]} 

grid_search = GridSearchCV(model, param_tunning, scoring='r2', cv=5, n_jobs=n_jobs)

logger.info('Starting grid search...')
grid_search.fit(X_train, y_train)

logger.info('Cross validating with best model...')
final_result = cross_validate(grid_search.best_estimator_, X_train, y_train, return_train_score=True, cv=cv, n_jobs=n_jobs)

train_score = final_result['train_score'].mean()
test_score = final_result['test_score'].mean()
assert train_score > 0.8
assert test_score > 0.7

logger.info(f'Train score: {train_score}')
logger.info(f'Test score: {test_score}')

logger.info('Updating model...')
update_model(grid_search.best_estimator_)

logger.info('Generating model report...')
validation_score = grid_search.best_estimator_.score(X_test, y_test)
save_simple_metrics_report(train_score, test_score, validation_score, grid_search.best_estimator_)

y_test_pred = grid_search.best_estimator_.predict(X_test)
get_model_performance_test_set(y_test, y_test_pred)

logger.info('Training finished!')