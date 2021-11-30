# coding=utf8
import logging
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataAnalyser:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(r'logs/exploratory_data_analysis.log', mode='w')
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        log_format = '%(name)s:%(levelname)s %(asctime)s -  %(message)s'
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        self.full_df = pd.read_csv('.csv files/full_df.csv', index_col=[0])
        columns_list = self.full_df.columns
        type_list = ['category', 'int16', 'int16', 'int16', 'int16', 'int32', 'int32', 'int32', 'float32']
        for column, ctype in zip(columns_list, type_list):
            self.full_df[column] = self.full_df[column].astype(ctype)
        self.buf1 = StringIO()

    def start(self):
        self.logger.debug('Starting Class')
        self.full_df_current_state(self.buf1)
        self.district_plot()
        self.describe_plot()
        self.correlation_matrix()
        self.logger.debug('Ending Class')

    def full_df_current_state(self, buf):
        self.logger.debug(f"Current full_df.head()\n{self.full_df.head()}")
        self.full_df.info(buf=buf)
        self.logger.debug(f"Current full_df.info()\n{buf.getvalue()}")
        self.logger.debug(f"Current full_df.describe()\n{self.full_df.describe()}")

    def district_plot_base(self, file_name, feat_name):
        self.logger.debug(f'Generating district_x_{file_name}.png')
        fig, ax = plt.subplots(figsize=(30, 18))
        sns.barplot(x='District', y=feat_name, data=self.full_df, ax=ax)
        plt.xticks(rotation=90)
        feat_median = self.full_df[feat_name].median()
        ax.axhline(feat_median, ls='--')
        ax.text(0, feat_median + self.full_df[feat_name].max() / 300, "Median line")
        plt.savefig(f'plots/dist_x_{file_name}.png')

    def district_plot(self):
        self.district_plot_base('area', 'Area (m²)')
        self.district_plot_base('value', 'Total Value (R$)')
        self.district_plot_base('valuepm', 'Value per m²')

    def create_describe_dataframe(self):
        self.desc_df = pd.DataFrame()
        base_df = self.full_df.drop('District', axis=1)
        for col in base_df.columns:
            self.desc_df[col] = self.full_df[col].describe()
        self.desc_df = self.desc_df.reset_index()
        self.desc_df = self.desc_df.drop(0)

    def show_values_on_bars(self, ax):
        for p in ax.patches:
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(x, y, value, ha="center")

    def describe_plot_base(self, file_name, feat_name):
        self.logger.debug(f'Generating describe_{file_name}.png')
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.barplot(x=self.desc_df.index, y=feat_name, data=self.desc_df, ax=ax)
        # Resets the index of desc_df and then takes the values from its column, so it can get the text from it
        x_label = self.desc_df.reset_index()['index']
        ax.set_xticklabels(x_label)
        self.show_values_on_bars(ax)
        plt.savefig(f'plots/describe_{file_name}.png')

    def describe_plot(self):
        self.logger.debug('Generating describe_df.png')
        self.create_describe_dataframe()
        name_list = ['area', 'bedrooms', 'bathrooms', 'garagecars', 'rent', 'condfee', 'value', 'valuepm']
        feat_list = self.desc_df.drop('index', axis=1).columns
        for name, feat in zip(name_list, feat_list):
            self.describe_plot_base(name, feat)

    def correlation_matrix(self):
        self.logger.debug('Generating feats_correlation_heatmap.png')
        sns.set_theme(style="white")
        corr = self.full_df.drop('District', axis=1).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.subplots(figsize=(15, 15))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, vmin=0, vmax=1,
                    annot=True, square=True, cbar_kws={"shrink": .6})
        plt.savefig(f'plots/feats_correlation_heatmap.png')
        # The correlation matrix shows the weakest correlation between features and rent is bedroom
        # This feature is a strong candidate to be dropped before the final model training
