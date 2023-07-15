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
    example_table = pd.read_csv('static/example.csv')
    st.dataframe(example_table)

    st.write('Search for site')
    ch = st.checkbox('Do you want to perform a query?', False)
    if ch:
        search_table = bq_search_query()
        search_table['choice'] = pd.Series()  # [0 for _ in range(len(search_table.index))])
        search_table.loc[0, 'choice'] = 1

        search_table = search_table.rename(columns={'sim_sum': 'similarity_keywords',
                                                    'full_content': 'content',
                                                    'pr': 'page_rank', 'time': 'response_time',
                                                    'size': 'file_size'})
        search_table = search_table[['choice', 'name', 'content', 'similarity_keywords',
                                     'page_rank', 'file_size', 'content_length', 'response_time',
                                     'Number_of_Keywords']]
        search_table.to_csv('static/search_table.csv')
    else:
        search_table = pd.read_csv('static/search_table.csv')
    # st.dataframe(search_table)
    edited_df = st.data_editor(search_table,
                               column_config={
                                   'content': st.column_config.TextColumn(
                                       width='large'),
                                   'choice': st.column_config.CheckboxColumn(
                                       help='Select for analysis',
                                   )

                               },
                               disabled=('content', 'similarity_keywords',
                                         'page_rank', 'file_size', 'content_length',
                                         'response_time',
                                         'Number_of_Keywords'),
                               hide_index=True)
    st.dataframe(edited_df)
    try:
        new_content = st.data_editor(search_table.loc[0, 'content'])
        new_pr = st.data_editor(search_table.loc[0, 'page_rank'], st.slider('page rank', 0, 1))

        st.table(edited_df[edited_df['choice'] == 1])
    except KeyError:
        st.write('Select one row')

    # with open('/static/data_all.dataframe' , 'wb') as f:
    #     data_from_pickle = pickle.load(f)
    # st.dataframe(data.head())
    # st.dataframe(data_from_pickle)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
