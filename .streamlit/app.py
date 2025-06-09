import streamlit as st
from streamlit_option_menu import option_menu

from pages.inference_page import inference_page
from pages.training_page import training_page
from pages.preprocessing_page import preprocessing_page
from pages.dashboard_page  import dashboard_page
from pages.upload_page import upload_page
from pages.monitoring_page import monitoring_page
from pages.reports_page import reports_page

def main():
    st.set_page_config(
        page_title="My ML App",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.markdown("## Навігація")
        page = option_menu(
            menu_title=None,
            options=[
                "Dashboard", "Upload", "Preprocessing",
                "Training", "Inference", "Monitoring", "Reports"
            ],
            icons=[
                "speedometer2", "cloud-upload", "gear",
                "book", "play-circle", "bar-chart-line", "file-earmark-text"
            ],
            menu_icon="list",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#282434"        # тёмный фон
                },
                "icon": {
                    "color": "#8ab4f8",                  # прежний цвет иконок
                    "font-size": "1.3rem"
                },
                "nav-link": {
                    "font-size": "1rem",
                    "font-weight": "500",                # medium вместо bold
                    "text-align": "left",
                    "padding": "12px 16px",              # внутренние отступы
                    "margin-bottom": "8px",              # расстояние между пунктами
                    "color": "#c6c6c6",
                    "--hover-color": "#3a3a4a"
                },
                "nav-link-selected": {
                    "background-color": "#3a3a4a",
                    "color": "#ffffff",
                    "font-weight": "500"                 # medium
                },
            }
        )

    # Рендер страниц
    if page == "Dashboard":
        dashboard_page()
    elif page == "Upload":
        upload_page()
    elif page == "Preprocessing":
        preprocessing_page()
    elif page == "Training":
        training_page()
    elif page == "Inference":
        inference_page()
    elif page == "Monitoring":
        monitoring_page()
    elif page == "Reports":
        reports_page()

if __name__ == "__main__":
    main()
