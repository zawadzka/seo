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


def bq_base_query(bq_table_name: str = bq_table):
    s = """
         SELECT name,  full_content, sim_sum as similarity, Number_of_Keywords 
         FROM {} where Number_of_Keywords between 20 
         and 100 and url like '%/site/%'
         and name is not null 
         limit 10
     """
    sql = s.format(bq_table_name)
    data = client.query(sql).to_dataframe()
    return data


def bq_search_query(bq_table_name: str = bq_table):
    search_name = st.text_input("Find coupon page for: ", "CCleaner")
    s = """
    SELECT name,  full_content, sim_sum, content_length, pr, 
    time, size, Number_of_Keywords 
    FROM {} where url like '%/site/%'
    and name like '%{}%'
    limit 10
    """
    sql = s.format(bq_table_name, search_name)
    data = client.query(sql).to_dataframe()
    return data


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.

    st.write(f'Hi {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    example_table = bq_base_query()
    st.dataframe(example_table)

    st.write('Search for site')
    search_table = bq_search_query()
    # search_table['choice'] = 0  # [lambda x: False for x in range(len(search_table))]
    search_table.columns = ['name', 'content', 'similarity to keyword set',
                            'content length', 'Page rank', 'response time',
                            'file size', 'number of keywords']

    st.dataframe(search_table,
                 column_config={
                     'content': st.column_config.TextColumn(
                         width='large'),
                     'name': st.column_config.SelectboxColumn(

                         help='Select for analise',
                         default=0)

                 },
                 hide_index=True)
    # selected_row = search_table[search_table['name'] is True]

    # with open('/static/data_all.dataframe' , 'wb') as f:
    #     data_from_pickle = pickle.load(f)
    # st.dataframe(data.head())
    # st.dataframe(data_from_pickle)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
