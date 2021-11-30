# coding=utf8
import streamlit as st
import pandas as pd
import numpy as np
from final_model_trainer import ModelTrainer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class WebApp:

    def __init__(self):
        pd.options.display.max_columns = 100
        pd.options.display.max_rows = 500
        pd.options.display.width = 0
        pd.options.display.float_format = '{:,.2f}'.format
        self.full_df = pd.read_csv('.csv files/full_df.csv', index_col=[0])
        self.model = ModelTrainer()
        self.model.web_model()
        self.le = LabelEncoder()

    def start(self):
        self.df_type_changer()
        self.title()
        self.user_input_features()
        self.create_predict_df()
        self.predict_price()
        self.create_valperm2_feat_df()
        self.create_boxplot()

    def df_type_changer_base(self, df, type_list):
        col_list = df.columns
        for column, ctype in zip(col_list, type_list):
            df[column] = df[column].astype(ctype)

    def df_type_changer(self):
        self.df_type_changer_base(self.full_df,
                                  ['str', 'int16', 'int16', 'int16', 'int16',
                                   'int32', 'int32', 'int32', 'float32'])

    def title(self):
        st.write("""
        
        # Calculadora de preço de aluguel de apartamento em São Paulo
        
        
        A calculadora a seguir usa informações coletadas de 10 mil anúncios de aluguel na cidade de São Paulo, de Novembro de
        2021, para estimar os valores de outros apartamentos da cidade, baseado em características como bairro e tamanho do
        apartamento.
        
        """)
        st.write("#")

    def user_input_features(self):
        with st.form(key='my-form'):

            st.subheader('Características do apartamento')

            district = st.selectbox(
                "Escolha o Bairro em que seu apartamento se encontra.\n Caso o seu Bairro não se encontre na lista, "
                "escolha a opção 'Outro'",
                ('Outro', 'Aclimação', 'Alto Da Boa Vista', 'Alto da Lapa', 'Alto de Pinheiros', 'Barra Funda', 'Bela Vista',
                 'Bom Retiro', 'Brooklin', 'Brás', 'Butantã', 'Cambuci', 'Campo Belo', 'Campos Eliseos', 'Casa Verde',
                 'Centro', 'Cerqueira César', 'Chácara Inglesa', 'Chácara Klabin', 'Chácara Santo Antônio',
                 'Cidade Monções', 'Consolação', 'Freguesia do Ó', 'Granja Julieta', 'Higienópolis', 'Indianópolis',
                 'Ipiranga', 'Itaim Bibi', 'Jabaquara', 'Jaguaré', 'Jardim América', 'Jardim Anália Franco',
                 'Jardim Europa', 'Jardim Marajoara', 'Jardim Paulista', 'Jardim Prudência', 'Jardins', 'Lapa', 'Liberdade',
                 'Mirandópolis', 'Moema', 'Morumbi', 'Móoca', 'Panamby', 'Paraíso', 'Perdizes', 'Pinheiros',
                 'Pirituba', 'Pompeia', 'República', 'Santa Cecília', 'Santa Ifigênia', 'Santana', 'Santo Amaro', 'Saúde',
                 'Sumaré', 'Tatuapé', 'Vila Anastácio', 'Vila Andrade', 'Vila Buarque', 'Vila Carrão', 'Vila Clementino',
                 'Vila Cruzeiro', 'Vila Formosa', 'Vila Gomes Cardim', 'Vila Guarani', 'Vila Leopoldina', 'Vila Madalena',
                 'Vila Mariana', 'Vila Mascote', 'Vila Nova Conceição', 'Vila Olímpia', 'Vila Prudente', 'Vila Romana',
                 'Vila Santa Catarina', 'Vila Sônia', 'Vila Uberabinha', 'Água Branca')
            )
            area = st.number_input('Insira a ÁREA do apartamento (em m²)', min_value=10, step=1)
            bedrooms = st.number_input('Insira o NÚMERO DE QUARTOS do apartamento', min_value=1, step=1)
            bathrooms = st.number_input('Insira o NÚMERO DE BANHEIROS do apartamento', min_value=1, step=1)
            garage_cars = st.number_input('Insira o NÚMERO DE VAGAS DE GARAGEM do apartamento', min_value=0, step=1)
            cond_fee = st.number_input('Insira o VALOR DO CONDOMÍNIO do apartamento (em R$)', min_value=0, step=1)

            st.form_submit_button('Calcular')

        data = {'District': district,
                'Area (m²)': area,
                'Bedrooms': bedrooms,
                'Bathrooms': bathrooms,
                'Garage Cars': garage_cars,
                'Condominium Fee (R$)': cond_fee}
        self.features_df = pd.DataFrame(data, index=[0])


    def create_predict_df(self):
        self.predict_df = self.features_df.drop(['Bedrooms', 'Bathrooms', 'Garage Cars'], axis=1)
        self.le.fit(self.full_df['District'])
        self.predict_df['District'] = self.le.transform(self.predict_df['District'])

    def predict_price(self):
        self.prediction = self.model.forest.predict(self.predict_df)
        self.prediction = np.around(self.prediction, 2)
        st.write("#")
        st.write(f'#### O valor estimado do aluguel é: **{self.prediction.item(0)} R$**')
        st.write("#")

    def create_valperm2_feat_df(self):
        self.features_df['Value per m²'] = self.prediction / self.features_df['Area (m²)']

    def create_boxplot_base(self, feature, feat_name, hline):
        plot_df = self.full_df.copy()
        plot_df['District'] = 'Cidade de São Paulo'
        plot_df = plot_df.append(self.full_df.loc[self.full_df['District'].isin(self.features_df['District'].values)])
        fig, ax = plt.subplots()
        sns.boxplot(x='District', y=feature, data=plot_df, ax=ax, showfliers=False,
                    order=[self.features_df['District'].iloc[0], 'Cidade de São Paulo'])
        ax.axhline(hline, ls='--')
        ax.text(-0.5, hline, "Valor informado")
        ax.set(xlabel='Local', ylabel=feat_name)
        st.pyplot(fig)

    def create_boxplot(self):
        st.write('### Agora, alguns dados sobre o seu apartamento em comparação a outros do mesmo bairro e de toda a cidade de São Paulo ')
        st.write('Para uma rápida explicação sobre como interpretar os gráficos abaixo, '
                 'clique [neste link](https://pt.wikipedia.org/wiki/Diagrama_de_caixa)')
        st.write('#### Comparação de Aluguel por Local')
        self.create_boxplot_base('Rent (R$)', 'Aluguel (R$)', self.prediction)
        st.write("#")
        st.write("#")
        st.write('#### Comparação de Área por Local')
        self.create_boxplot_base('Area (m²)', 'Área (m²)', self.features_df['Area (m²)'].iloc[0])
        st.write("#")
        st.write("#")
        st.write('#### Comparação de Taxa de Condomínio por Local')
        self.create_boxplot_base('Condominium Fee (R$)', 'Condomínio (R$)',
                                 self.features_df['Condominium Fee (R$)'].iloc[0])
        st.write("#")
        st.write("#")
        st.write('#### Comparação de Valor de Metro Quadrado por Local')
        self.create_boxplot_base('Value per m²', 'Valor por m²', self.features_df['Value per m²'].iloc[0])
        st.write("#")
        st.write("#")
        st.write("#")
        st.write('Obrigado por utilizar a minha calculadora :) -- Danilo Rosmaninho')



web_app = WebApp()
web_app.start()
