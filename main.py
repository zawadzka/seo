import os.path

import streamlit as st
import pandas as pd
import numpy as np
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


@st.cache_data
def bq_search_query(search_name, bq_table_name: str = bq_table):
    s = """
    SELECT name,  full_content, sim_sum, content_length, pr, 
    time, size, Number_of_Keywords 
    FROM {} where url like '%/site/%'
    and url like '%{}%'
    limit 10
    """
    sql = s.format(bq_table_name, search_name)
    data = client.query(sql).to_dataframe()

    return data


def main():
    # Use a breakpoint in the code line below to debug your script.

    st.write(f'''Data about pages consist of company/product name, 
                editorial content written about this company, 
                and some descriptive fields - calculated inner page rank, 
                semantic similarity to typical coupon 
                and promotion related keywords, 
                file size, page load speed - response time during crawling, 
                and number of keywords the company's coupon page is visible for 
                (based on an external tool).  
                ''')
    st.write('The table beneath show some example with basic fields')
    example_table = pd.read_csv('static/example.csv')
    st.dataframe(example_table)

    st.write('Search for site')

    def load():
        q = st.text_input('company name', 'FlixBus')

        q = q.strip().lower().replace(r'\s+', '-')
        try:
            s_table = bq_search_query(q)
            s_table['choice'] = pd.Series([x for x in range(len(s_table))])

            s_table = s_table.rename(columns={'sim_sum': 'similarity_keywords',
                                                        'full_content': 'content', 'time': 'response_time',
                                                        'size': 'file_size'})
            s_table = s_table[['choice', 'name', 'content', 'similarity_keywords',
                                         'pr', 'file_size', 'content_length', 'response_time',
                                         'Number_of_Keywords']]

        except KeyError:
            st.write(f'There is no {q} in the database. Example table:')
            s_table = pd.read_csv('static/search_table.csv')
        if not s_table.empty:
            s_table.to_csv('static/search_table.csv')
        # st.table(search_table)
        return s_table

    search_table = load()
    st.dataframe(search_table)

    try:
        selected_indices = st.multiselect('Select one row:', search_table.name,
                                      default=search_table.name[0], max_selections=1)
    except (ValueError, KeyError):
        st.write(f'There is no such query in the database. Example table:')
        search_table = pd.read_csv('static/search_table.csv')
        selected_indices = st.multiselect('Select one row:', search_table.name,
                                      default=search_table.name[0], max_selections=1)
    try:
        ind = search_table[search_table['name'] == selected_indices[0]].index[0]
        content = search_table.loc[ind, 'content']
        pr_v = float(search_table.loc[ind, 'pr'])
    except IndexError:
        content = search_table.loc[0, 'content']
        pr_v = float(search_table.loc[0, 'pr'])

    new_content = st.text_area('Change text to examine new content', content)
    new_pr = st.slider('Insert new page rank value', 0.0, 1.0, pr_v, step=0.01)
    st.write(new_pr, new_content)

    # try:
    # ind = search_table[search_table['name'==selected_indices[0]].index[0]
    #
    #     pr_v = search_table.loc[ind, 'pr']
    #     content = edited_df.loc[ind[0][0], 'content']
    # except KeyError:
    #     st.write('Select one row')
    # try:
    #     new_content = st.text_area('new content', content)
    #     new_pr = st.slider('page rank', 0.0, 1.0, pr_v)
    # except KeyError:

    #     new_content = content
    # st.write(new_content)

    # st.dataframe(search_table)
    # edited_df = st.data_editor(search_table,
    #                            column_config={
    #                                'content': st.column_config.TextColumn(
    #                                    width='large'),
    #                                'choice': st.column_config.CheckboxColumn(
    #                                    help='Select for analysis',
    #                                )
    #
    #                            },
    #                            disabled=('similarity_keywords',
    #                                      'Number_of_Keywords'),
    #                            hide_index=True)
    #
    # ind = np.where(edited_df['choice'].to_numpy() == 1)

    # st.write(new_content)
    # with open('/static/data_all.dataframe' , 'wb') as f:
    #     data_from_pickle = pickle.load(f)
    # st.dataframe(data.head())
    # st.dataframe(data_from_pickle)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
