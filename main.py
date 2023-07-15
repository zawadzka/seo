import streamlit as st
# from google.oauth2 import service_account
# from google.cloud.exceptions import NotFound
# from google.cloud import bigquery

# credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])


# secrets = toml.load('secrets.toml')["gcp_service_account"]
# project_id = 'seo-project-392909'
# secrets = st.secrets["gcp_service_account"]
# credentials = service_account.Credentials.from_service_account_info(secrets)
# client = bigquery.Client(credentials=credentials)
# client = bigquery.Client(credentials=credentials, project=project_id)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    name = st.text_input('name')
    st.write(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
