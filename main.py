import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import utils

credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
# secrets = toml.load('secrets.toml')["gcp_service_account"]
project_id = 'seo-project-392909'
secrets = st.secrets["gcp_service_account"]
client = bigquery.Client(credentials=credentials, project=project_id)
bq_table = 'seo-project-392909.seo_dataset.data'


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
                                              'full_content': 'content',
                                              'time': 'response_time',
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
    except IndexError:
        ind = 0

    content = search_table.loc[ind, 'content']
    pr_v = float(search_table.loc[ind, 'pr'])
    name = search_table.loc[ind, 'name']
    size = search_table.loc[ind, 'file_size']
    time = search_table.loc[ind, 'response_time']

    with st.form("Try new values!"):
        new_content = st.text_area('Change text to examine new content', content)
        new_pr = st.slider('Insert new page rank value', 0.0, 0.01, pr_v, step=0.0001)
        size_divided = size//10000
        new_size = st.slider('Insert new page size value - x 10k', 1, int(size_divided), 10, step=1)*10000
        new_time = st.slider('Insert new page time value', 1.0, 2.0, float(time), step=0.1)

        submitted = st.form_submit_button('Calculate predictions')
        if submitted:
            st.write(f'New content: {new_content}')
            st.write(f'New page rank: {new_pr}')

            st.write(f'time: {new_time}\n size: {new_size}')

    page = utils.InputData(new_content, name, new_pr, new_size, new_time)
    ss_rounded = '{:.2f}'.format(page.sim_sum)
    st.write(f"similarity: {ss_rounded}\n content length: {page.content_length}")
    y = utils.make_prediction(page)
    st.write(f'predicted: {y}')
    if y > 0.5:
        st.write(':sparkles: Good, predicted visibility better than average.')
    else:
        st.write(':disappointed:, predicted visibility worse than average.')


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
