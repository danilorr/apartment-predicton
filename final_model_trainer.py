import logging
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse


class ModelTrainer:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler('logs/final_model_trainer.log',
                                           mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        self.train = pd.read_csv('.csv files/train.csv', index_col=[0])
        self.test = pd.read_csv('.csv files/test.csv', index_col=[0])
        self.test_leakage = pd.read_csv('.csv files/test_leakage.csv', index_col=[0])
        df_list = [self.train, self.test]
        for df in df_list:
            columns_list = df.columns
            type_list = ['int8', 'int16', 'int16', 'int16', 'int16', 'int32', 'int32', 'int32', 'float32']
            for column, ctype in zip(columns_list, type_list):
                df[column] = df[column].astype(ctype)
            df.drop(['Bathrooms', 'Bedrooms', 'Garage Cars'], axis=1, inplace=True)
        self.test_leakage['Rent (R$)'] = self.test_leakage['Rent (R$)'].astype('int32')
        self.forest = self.forest = RandomForestRegressor(random_state=42)
        self.buf1 = StringIO()
        self.buf2 = StringIO()
        self.buf3 = StringIO()


    def start(self):
        self.logger.debug('Starting Class')
        self.df_current_state(self.train, self.buf1, 'train')
        self.df_current_state(self.test, self.buf2, 'test')
        self.df_current_state(self.test_leakage, self.buf3, 'test_leakage')
        self.create_x_and_y_dfs()
        self.train_model()
        self.make_prediction()
        self.create_prediction_csv()
        self.model_scorer()
        self.logger.debug('Ending Class')

    def df_current_state(self, df, buf, name):
        self.logger.debug(f"Current {name}.head()\n{df.head()}")
        df.info(buf=buf)
        self.logger.debug(f"Current {name}.info()\n{buf.getvalue()}")

    def create_x_and_y_dfs(self):
        self.logger.debug('Creating X and y dataframes')
        self.y = self.train['Rent (R$)']
        self.X = self.train.drop(['Rent (R$)'], axis=1)

    def train_model(self):
        self.logger.debug('Training model')
        self.forest = RandomForestRegressor(
            max_depth=89,
            max_leaf_nodes=859,
            min_samples_split=185,
            n_estimators=710)
        self.forest.fit(self.X, self.y)

    def make_prediction(self):
        self.logger.debug('Predicting test df')
        self.y_pred = self.forest.predict(self.test)
        self.y_pred = np.around(self.y_pred)
        self.y_pred = self.y_pred.astype('int32')

    def create_prediction_csv(self):
        self.logger.debug('Creating test_leakage_prediction.csv')
        prediction = pd.DataFrame({'Rent (R$)': self.y_pred})
        prediction.to_csv('C:/Users/Windows/PycharmProjects/VivaReal/.csv files/test_leakage_prediction.csv')

    def model_scorer(self):
        score = mse(self.y_pred, self.test_leakage['Rent (R$)'], squared=False)
        self.logger.debug(f'The Random Forest final score was: {score}')

    def web_model(self):
        self.create_x_and_y_dfs()
        self.train_model()
