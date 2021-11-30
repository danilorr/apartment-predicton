import logging
from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBRegressor
from xgboost import plot_importance


class ModelSelector:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/model_selector.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        self.train = pd.read_csv('.csv files/train.csv', index_col=[0])
        self.test = pd.read_csv('.csv files/test.csv', index_col=[0])
        df_list = [self.train, self.test]
        for df in df_list:
            columns_list = df.columns
            type_list = ['int8', 'int16', 'int16', 'int16', 'int16', 'int32', 'int32', 'int32', 'float32']
            for column, ctype in zip(columns_list, type_list):
                df[column] = df[column].astype(ctype)
        self.scorer = make_scorer(mse)
        self.linreg = LinearRegression()
        self.elnet = ElasticNet(random_state=42)
        self.dectree = DecisionTreeRegressor(random_state=42)
        self.forest = RandomForestRegressor(random_state=42)
        self.adab = AdaBoostRegressor(random_state=42)
        self.xgb = XGBRegressor(eval_metric=self.scorer, verbosity=0, random_state=42)
        self.score_dic = {}
        self.buf1 = StringIO()
        self.buf2 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.train, self.buf1, 'train')
        self.df_current_state(self.test, self.buf2, 'test')
        self.create_x_and_y_dfs()
        self.make_train_test_split()
        self.feature_importance_xgboost()
        self.plot_feature_importance()
        self.drop_low_importance_features()
        self.test_linear_regression()
        self.test_models()
        self.test_xgboost()
        self.select_model()
        self.logger.debug('Ending Class')

    def df_current_state(self, df, buf, name):
        self.logger.debug(f"Current {name}.head()\n{df.head()}")
        df.info(buf=buf)
        self.logger.debug(f"Current {name}.info()\n{buf.getvalue()}")

    def create_x_and_y_dfs(self):
        self.logger.debug('Creating X and y dataframes')
        self.y = self.train['Rent (R$)']
        self.X = self.train.drop(['Rent (R$)'], axis=1)

    def make_train_test_split(self):
        self.logger.debug('Creating train test split')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

    def bayes_search(self, model, param_grid, name):
        self.logger.debug('Initializing Bayes Search')
        n_iter = 5
        cv = KFold(n_splits=n_iter, shuffle=True, random_state=42)
        bsearch = BayesSearchCV(
            model, param_grid, n_iter=n_iter, scoring=self.scorer, cv=cv, n_jobs=-1) \
            .fit(self.X_train, self.y_train)
        self.logger.debug(f'The best parameters for the {name} are:\n{bsearch.best_params_}')
        # The dictionary is saved as a list to later use it on the models
        self.param_attributes = list(bsearch.best_params_.keys())
        self.param_values = list(bsearch.best_params_.values())

    def feature_importance_xgboost(self):
        self.logger.debug('Creating the xgboost model for feature importance check')
        param_grid = {'max_depth': Integer(1, 90),
                      'learning_rate': Real(0.01, 1, prior='log-uniform'),
                      'reg_alpha': Real(0.01, 100),
                      'colsample_bytree': Real(0.2e0, 0.8e0),
                      'subsample': Real(0.2e0, 0.8e0),
                      'n_estimators': Integer(50, 200)}
        self.bayes_search(self.xgb, param_grid, 'xgboost')
        # The parameters are saved on a different variable so it can be used later on another xgboost fitting
        self.xgboost_attributes = self.param_attributes
        self.xgboost_values = self.param_values
        for attribute, value in zip(self.param_attributes, self.param_values):
            self.xgb.attribute = value
        self.xgb.fit(self.X_train,
                     self.y_train,
                     eval_metric="rmse",
                     eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                     early_stopping_rounds=20)

    def plot_feature_importance(self):
        self.logger.debug('Plotting features')
        fig, ax = plt.subplots(figsize=(10, 15))
        fig1 = plt.gcf()
        plot_importance(booster=self.xgb, ax=ax)
        plt.draw()
        fig1.savefig('plots/feature_importance.png')
        # The feature importance plot shows a really low importance on Bedrooms, Bathrooms and Car Garages
        # To help the other models, these features will be dropped

    def drop_low_importance_features(self):
        self.logger.debug('Dropping low importance features')
        feat_list = ['Bathrooms', 'Bedrooms', 'Garage Cars']
        self.X = self.X.drop(feat_list, axis=1)
        self.X_train = self.X_train.drop(feat_list, axis=1)
        self.X_test = self.X_test.drop(feat_list, axis=1)

    def model_scorer(self, model, name):
        y_pred = model.predict(self.X_test)
        score = mse(y_pred, self.y_test, squared=False)
        # The score of each model is saved for later comparison
        self.score_dic[name] = score
        self.logger.debug(f'The {name} score was: {score}')

    def test_linear_regression(self):
        # Linear regression is tested in a separated method since it does not need hyperparameter search
        self.logger.debug('Testing the linear regression model')
        self.linreg.fit(self.X_train, self.y_train)
        self.model_scorer(self.linreg, 'Linear Regressor')

    def test_model_base(self, model, param_grid, name):
        self.logger.debug(f'Testing the {name} model')
        self.bayes_search(model, param_grid, name)
        for attribute, value in zip(self.param_attributes, self.param_values):
            model.attribute = value
        model.fit(self.X_train, self.y_train)
        self.model_scorer(model, name)

    def test_models(self):
        param_grid = {'alpha': Real(0.1, 1, prior='log-uniform'),
                      'l1_ratio': Real(0, 1),
                      'max_iter': Integer(50, 5000)}
        self.test_model_base(self.elnet, param_grid, 'Elastic Net')

        param_grid = {'criterion': Categorical(['squared_error', 'poisson']),
                      'splitter': Categorical(['best', 'random']),
                      'max_depth': Integer(20, 2000),
                      'min_samples_split': Integer(2, 200),
                      'max_leaf_nodes': Integer(10, 1000)}
        self.test_model_base(self.dectree, param_grid, 'Decision Tree')

        param_grid = {'n_estimators': Integer(100, 5000),
                      'criterion': Categorical(['squared_error', 'poisson']),
                      'max_depth': Integer(20, 2000),
                      'min_samples_split': Integer(2, 200),
                      'max_leaf_nodes': Integer(10, 1000)}
        self.test_model_base(self.forest, param_grid, 'Random Forest')

        param_grid = {'n_estimators': Integer(100, 5000),
                      'learning_rate': Real(0.01, 1, prior='log-uniform'),
                      'loss': Categorical(['linear', 'square', 'exponential'])}
        self.test_model_base(self.adab, param_grid, 'AdaBoost')

    def test_xgboost(self):
        # The xgboost is tested in a separated method since its fitting arguments are unique
        self.logger.debug('Testing the xgboost model')
        for attribute, value in zip(self.xgboost_attributes, self.xgboost_values):
            self.xgb.attribute = value
        self.xgb.fit(self.X_train,
                     self.y_train,
                     eval_metric="rmse",
                     eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                     early_stopping_rounds=20)
        self.model_scorer(self.xgb, 'XGBoost')

    def select_model(self):
        self.logger.debug('Finding the best scoring model')
        model_names = list(self.score_dic.keys())
        model_scores = list(self.score_dic.values())
        best_model_index = model_scores.index(min(model_scores))
        self.logger.debug(f'The best model is: {model_names[best_model_index]}')
