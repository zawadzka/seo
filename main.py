import os.path

import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import pickle


credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
# secrets = toml.load('secrets.toml')["gcp_service_account"]
project_id = 'seo-project-392909'
secrets = st.secrets["gcp_service_account"]
# credentials = service_account.Credentials.from_service_account_info(secrets)
# client = bigquery.Client(credentials=credentials)
client = bigquery.Client(credentials=credentials, project=project_id)
bq_table = 'seo-project-392909.seo_dataset.data'


# bq_secrets = secrets, bq_client = client,


def bq_import(bq_table_name: str = bq_table):
    s = """
         SELECT sim_sum as similarity, full_content, Number_of_Keywords as n_keywords
         FROM {} where n_keywords<20 limit 10
     """
    sql = s.format(bq_table_name)
    data = client.query(sql).to_dataframe()
    return data


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.

    st.write(f'Hi {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    data = bq_import()
    st.dataframe(data)
    # with open('/static/data_all.dataframe' , 'wb') as f:
    #     data_from_pickle = pickle.load(f)
    # st.dataframe(data.head())
    # st.dataframe(data_from_pickle)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
