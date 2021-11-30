import logging
from io import StringIO
import pandas as pd
from fuzzywuzzy import process
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataProcessor:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/data_processor.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.le = LabelEncoder()
        self.full_df = pd.read_csv('.csv files/apts_dataframe.csv')
        self.buf1 = StringIO()
        self.buf2 = StringIO()
        self.buf3 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.full_df_current_state(self.buf1)
        self.check_title_relevance()
        self.convert_string_columns_into_integer()
        self.turn_address_feature_into_district()
        self.rename_full_df_columns()
        self.full_df_current_state(self.buf2)
        self.district_string_similarity_test()
        self.group_less_frequent_districts()
        self.create_total_value_features()
        self.drop_features_outliers()
        self.create_full_df_csv()
        self.full_df_current_state(self.buf3)
        self.district_encoder()
        self.create_x_and_y_splits()
        self.create_test_leakage_df_csv()
        self.create_test_df_csv()
        self.create_train_df_csv()
        self.logger.debug('Ending Class')

    def full_df_current_state(self, buf):
        self.logger.debug(f"Current full_df.head()\n{self.full_df.head()}")
        self.full_df.info(buf=buf)
        self.logger.debug(f"Current full_df.info()\n{buf.getvalue()}")

    def check_title_relevance(self):
        self.logger.debug(f"Checking frequency of Aluguel in rows: "
                          # This line of code sums all rows that contains Aluguel in its Title
                          f"{self.full_df['Title'].str.contains('Aluguel', regex=False).sum()}")
        self.logger.debug('Dropping Title Column from full_df')
        self.full_df = self.full_df.drop(['Unnamed: 0', 'Title'], axis=1)

    def string_to_integer_basic_converter(self, feat_list):
        self.full_df.loc[self.full_df['Garage Cars'] == '-- Vaga', 'Garage Cars'] = '0 Vaga'
        for feat in feat_list:
            # This line takes the 1st split if the string by ' '
            self.full_df[feat] = self.full_df[feat].str.split(' ').str[0]
            self.full_df[feat] = self.full_df[feat].astype('int16')

    def string_to_integer_money_converter(self, feat_list):
        for feat in feat_list:
            self.full_df[feat] = self.full_df[feat].str.replace('.', '', regex=False)
            # This line takes the 2nd split if the string by ' '
            self.full_df[feat] = self.full_df[feat].str.split(' ').str[1]
            self.full_df[feat] = self.full_df[feat].astype('int32')

    def convert_string_columns_into_integer(self):
        self.logger.debug('Converting string columns into integer')
        feat_list = ['Area', 'Bedrooms', 'Bathrooms', 'Garage Cars']
        self.string_to_integer_basic_converter(feat_list)
        feat_list = ['Rent', 'Condominium']
        self.string_to_integer_money_converter(feat_list)

    def fix_address_outliers(self):
        self.logger.debug('Fixing Address Outliers')
        # Taking the Address list of all rows that doesnt end with , São Paulo - SP
        # As all of them ends with '- SP' instead, erase that and append ', São Paulo - SP' to the end of the string
        self.full_df.loc[~self.full_df['Address'].str.endswith(', São Paulo - SP'), 'Address'] = \
            self.full_df['Address'].astype(str).str[0:-5] + ', São Paulo - SP'

    def extract_district_from_address(self):
        self.logger.debug('Extracting District from Address')
        # Select all Address rows that, after removing the last 16 characters, does not contain ' - '
        # For those, remove the last 16 characters
        self.full_df.loc[~self.full_df['Address'].str[0:-16].str.contains(' - '), 'Address'] = \
            self.full_df['Address'].str[0:-16]
        # Select all Address rows that, after removing the last 16 characters, still contains ' - '
        # For those, remove the last 16 characters, take the second split by '-' and remove the 1st character
        self.full_df.loc[self.full_df['Address'].str[0:-16].str.contains(' - '), 'Address'] = \
            self.full_df['Address'].str[0:-16].str.split('-').str[1].str[1:]

    def turn_address_feature_into_district(self):
        self.fix_address_outliers()
        self.extract_district_from_address()

    def rename_full_df_columns(self):
        self.logger.debug('Renaming full_df columns')
        self.full_df = self.full_df.rename(columns={'Address': 'District', 'Area': 'Area (m²)',
                                                    'Rent': 'Rent (R$)', 'Condominium': 'Condominium Fee (R$)'})

    def district_string_similarity_test(self):
        self.logger.debug('Checking the similarity among District strings')
        # Taking the index of the 10 most frequent districts
        most_frequent_districts = self.full_df['District'].value_counts().head(10).index
        district_list = self.full_df['District'].unique()
        for district in most_frequent_districts:
            # Comparing the 10 most frequent districts with the list of all unique districts
            self.logger.debug(process.extract(district, district_list)[0:2])
        # The actual comparison was made with a much larger number of districts than the 10 listed
        # But the list was shortened to 10 as an illustration since the more complete comparison
        # ended having the same result
        self.logger.debug('None of them is a misspelling of the other, therefore no corrections needed')

    def group_less_frequent_districts(self):
        self.logger.debug('Grouping the less frequent districts into a single category')
        # The bottom 403 districts were chosen to be grouped up
        other_district = self.full_df['District'].value_counts().tail(403).index
        for district in other_district:
            # Looping through all the bottom districts and renaming them as 'Outro'
            self.full_df.loc[self.full_df['District'] == district, 'District'] = 'Outro'

    def create_total_value_features(self):
        self.logger.debug('Creating Total Value and Value per m² features')
        self.full_df['Total Value (R$)'] = self.full_df['Rent (R$)'] + self.full_df['Condominium Fee (R$)']
        self.full_df['Value per m²'] = (self.full_df['Total Value (R$)'] / self.full_df['Area (m²)']).astype('float32')

    def drop_outliers(self, feature, drop_list):
        self.logger.debug(f"Looking into apartments with largest {feature} values through value_counts():\n"
                          f"{self.full_df[feature].value_counts().sort_index(ascending=False).head(10)}")
        self.logger.debug(f"Dropping the following item(s):\n{self.full_df[self.full_df.index.isin(drop_list)]}")
        self.full_df = self.full_df.drop(drop_list)

    def drop_features_outliers(self):
        self.drop_outliers('Area (m²)', [8556])
        self.drop_outliers('Bedrooms', [4771])
        self.drop_outliers('Garage Cars', [8559, 2442, 554, 2644, 8971])
        self.drop_outliers('Rent (R$)', [2756, 9854, 5046, 2025])
        self.drop_outliers('Condominium Fee (R$)', [5515, 9220, 7939, 551, 6957, 5294, 4715, 96, 9755, 1469, 6960])
        self.drop_outliers('Value per m²', [5457, 5079, 9368, 54, 7999, 6746, 8806,
                                            6180, 2684, 7539, 4615, 7415, 4300, 3837, 6477])

    def create_full_df_csv(self):
        self.logger.debug('Creating full_df.csv')
        # This csv will be used in the exploratory_data_analyser file
        self.full_df = self.full_df.reset_index(drop=True)
        self.full_df.to_csv(r'.csv files/full_df.csv')

    def district_encoder(self):
        self.logger.debug('Encoding district as ordinals')
        self.full_df['District'] = self.le.fit_transform(self.full_df['District'])

    def create_x_and_y_dfs(self):
        self.logger.debug('Creating X and y dataframes')
        self.y = self.full_df['Rent (R$)']
        self.X = self.full_df.drop(['Rent (R$)', 'Total Value (R$)', 'Value per m²'], axis=1)

    def create_split(self):
        self.logger.debug('Creating X and y train/test splits')
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def create_x_and_y_splits(self):
        self.create_x_and_y_dfs()
        self.create_split()

    def create_test_leakage_df_csv(self):
        self.logger.debug('Creating test_leakage.csv')
        # The test leakage will have all features that will be considered leakage in the model training
        self.test_leakage = self.y_test.reset_index(drop=True)
        self.test_leakage.to_csv(r'.csv files/test_leakage.csv')

    def create_test_df_csv(self):
        self.logger.debug('Creating test.csv')
        self.test = self.X_test.reset_index(drop=True)
        self.test.to_csv(r'.csv files/test.csv')

    def create_train_df_csv(self):
        self.logger.debug('Creating train.csv')
        self.train = self.X_train.join(self.y_train).reset_index(drop=True)
        self.train.to_csv(r'.csv files/train.csv')
