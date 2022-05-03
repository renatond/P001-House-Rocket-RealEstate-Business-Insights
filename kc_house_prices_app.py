import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import folium
import geopandas
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from PIL import Image

# =======================================================================
# Settings
# =======================================================================

# Geofile definition
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = geopandas.read_file(url)

# pandas config
pd.set_option('display.float_format', lambda x: '%.1f' % x)

# streamlit config
st.set_page_config(layout='wide')

# =======================================================================
# Load Data
# =======================================================================

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data

# =======================================================================
# Data transformation
# =======================================================================

# adjusting datetime format =============================================
def correct_datetimes(data):
    data['date'] = pd.to_datetime(data['date']).dt.date

    return None

# treatig outliers ======================================================
def treat_outliers(data):
    data = data.sort_values('bedrooms', ascending=False).reset_index()
    data.loc[0, 'bedrooms'] = 3

    return data

# convert areas to metric system ========================================
def convert_to_metric(data):
    data['m2_living'] = data['sqft_living'] * 0.092903
    data['m2_lot'] = data['sqft_lot'] * 0.092903
    data['m2_above'] = data['sqft_above'] * 0.092903
    data['m2_basement'] = data['sqft_basement'] * 0.092903
    data['price_m2'] = data['price']/(data['sqft_lot'] * 0.092903)

    data = data.drop(['sqft_living', 'sqft_living15',
                    'sqft_lot', 'sqft_lot15', 'sqft_above', 'sqft_basement'], axis=1)

    return data

# create new attributes
def create_new_attributes(data):
    # getting datetime info ===============================================
    data['date_year'] = pd.to_datetime(data['date']).dt.year
    data['date_month'] = pd.to_datetime(data['date']).dt.month
    data['date_week'] = pd.to_datetime(data['date']).dt.week

    # getting season ==================================================
    data['season'] = data['date_month'].apply( lambda x: 'Winter' if (x == 12 or x <= 2) else
                                                            'Spring' if (3 <= x < 6) else
                                                            'Summer' if (6 <= x <= 8) else 'Autumn')

    # Has Basement ===================================================== 
    data['basement'] = data['m2_basement'].apply( lambda x: 'Has Basement' if x != 0 else 'No Basement')

    # getting selling seasonality
    seasonality_df = data[['zipcode','season', 'price']].groupby(['zipcode', 'season']).mean().reset_index()
    seasonality_df = seasonality_df.loc[seasonality_df.groupby(['zipcode'])['price'].idxmax()].drop('price', axis=1)
    seasonality_df.columns = ['zipcode', 'seasonality']
    data = pd.merge(data, seasonality_df, on='zipcode', how='inner')

    return data

# create investment dataset =========================================
def create_investment_dataset(data):
    # create medians dataset ================================================
    attributes = ['price', 'bedrooms','bathrooms', 'm2_living', 'm2_lot', 'floors',
                'view', 'condition', 'grade', 'm2_above', 'm2_basement', 'zipcode']

    median_by_zipcode = data[attributes].groupby('zipcode').median().reset_index()
    median_by_zipcode.columns = ['zipcode', 'median_price', 'median_bedrooms','median_bathrooms', 'median_m2_living', 'median_m2_lot', 'median_floors',
                'median_view', 'median_condition', 'median_grade', 'median_m2_above', 'median_m2_basement']

    comparative_dataset = pd.merge(data, median_by_zipcode, on='zipcode', how='inner')
    investment_attr = ['id', 'date', 'bedrooms', 'bathrooms', 'floors', 'm2_living', 'm2_lot', 'waterfront', 'grade', 'zipcode', 'lat', 'long', 'price', 'selling_price', 'profit', 'margin' ]
    investment_dataset = comparative_dataset[(comparative_dataset['price'] < comparative_dataset['median_price']) &
            (comparative_dataset['condition'] > comparative_dataset['median_condition']) &
            (comparative_dataset['m2_lot'] > comparative_dataset['median_m2_lot']) &
            (comparative_dataset['m2_living'] > comparative_dataset['median_m2_living'])].reset_index().drop('index', axis=1)

    investment_dataset['selling_price'] = investment_dataset['median_price'] * 1.3
    investment_dataset['profit'] = investment_dataset['selling_price'] - investment_dataset['price']
    investment_dataset['margin'] = investment_dataset['profit']/investment_dataset['selling_price']

    investment_dataset = investment_dataset[investment_attr]

    return investment_dataset

# Measures ============================================================
def create_measures(data):
    yr_built_list = data['yr_built'].unique()
    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())
    min_price = int( data['price'].min() )
    max_price = int( data['price'].max() )
    avg_price = int(data['price'].mean())
    median_price = int(data['price'].median())
    min_bedrooms = data['bedrooms'].min()
    max_bedrooms = data['bedrooms'].max()
    min_floors = data['floors'].min()
    max_floors = data['floors'].max()
    min_bathrooms = data['bathrooms'].min()
    max_bathrooms = data['bathrooms'].max()

    return None

# =========================================================================
# # Sidebar filters
# =========================================================================
def sidebar_filters(data):
    st.sidebar.title('Dataset Filters')

    # Data Overview Filters
    data_overview_filters = st.sidebar.expander(label='Data Overview Filters')
    with data_overview_filters:
       # dataset columns ===================================================================================================
        data_selection = data.columns.tolist()
        data_selection.append('ALL')
        default_att = ['id', 'date', 'yr_built', 'price', 'bedrooms','bathrooms', 'm2_living', 'm2_lot', 'zipcode']
        count = 1

        columns_containter = data_overview_filters.container()
        all_columns = data_overview_filters.checkbox('Select all columns', value=False, key=count)
        count += 1

        if all_columns:
            s_attributes = columns_containter.multiselect("Select one or more options:", data_selection, default_att)
            f_attributes = data.columns.tolist()
        else:
            s_attributes = columns_containter.multiselect("Select one or more options:", data_selection, default_att)
            if 'ALL' in s_attributes:
                f_attributes = data.columns.tolist()
            else:
                f_attributes = s_attributes

        # zipcode selection ==================================================================================================
        zipcode_selection = data['zipcode'].unique().tolist()
        zipcode_selection.append('ALL')

        zipcode_container = data_overview_filters.container()
        all_zipcodes = data_overview_filters.checkbox('Select all zipcodes', value=True, key=count)
        count += 1

        if all_zipcodes:
            s_zipcodes = zipcode_container.multiselect('Select zipcodes:', zipcode_selection, 'ALL')
            f_zipcodes = data['zipcode'].unique().tolist()
        else:
            s_zipcodes = zipcode_container.multiselect("Select one or more options:", zipcode_selection, 'ALL')
            if 'ALL' in s_zipcodes:
                f_zipcodes = data['zipcode'].unique().tolist()
            else:
                f_zipcodes = s_zipcodes

        # number of bedrooms ================================================================================================
        bedrooms_selection = sorted(data['bedrooms'].unique().tolist())
        bedrooms_selection.append('ALL')

        bedrooms_container = data_overview_filters.container()
        all_bedrooms = data_overview_filters.checkbox('Select all bedrooms', value=True, key=count)
        count += 1

        if all_bedrooms:
            s_bedrooms = bedrooms_container.multiselect('Select the numbers of bedrooms:', bedrooms_selection, 'ALL')
            f_bedrooms = sorted(data['bedrooms'].unique().tolist())
        else:
            s_bedrooms = bedrooms_container.multiselect('Select the numbers of bedrooms:', bedrooms_selection, 'ALL')
            if 'ALL' in s_bedrooms:
                f_bedrooms = sorted(data['bedrooms'].unique().tolist())
            else:
                f_bedrooms = s_bedrooms

        # number of bathrooms ============================
        bathrooms_selection = sorted(data['bathrooms'].unique().tolist())
        bathrooms_selection.append('ALL')

        bathrooms_container = data_overview_filters.container()
        all_bathrooms = data_overview_filters.checkbox('Select all bathrooms', value=True, key=count)
        count += 1

        if all_bathrooms:
            s_bathrooms = bathrooms_container.multiselect('Select the numbers of bathrooms:', bathrooms_selection, 'ALL')
            f_bathrooms = sorted(data['bathrooms'].unique().tolist())
        else:
            s_bathrooms = bathrooms_container.multiselect('Select the numbers of bathrooms:', bathrooms_selection, 'ALL')
            if 'ALL' in s_bathrooms:
                f_bathrooms = sorted(data['bathrooms'].unique().tolist())
            else:
                f_bathrooms = s_bathrooms

        # number of floors ============================
        floors_selection = sorted(data['floors'].unique().tolist())
        floors_selection.append('ALL')

        floors_container = data_overview_filters.container()
        all_floors = data_overview_filters.checkbox('Select all floors', value=True, key=count)
        count += 1

        if all_floors:
            s_floors = floors_container.multiselect('Select the numbers of floors:', floors_selection, 'ALL')
            f_floors = sorted(data['floors'].unique().tolist())
        else:
            s_floors = floors_container.multiselect('Select the numbers of floors:', floors_selection, 'ALL')
            if 'ALL' in s_floors:
                f_floors = sorted(data['floors'].unique().tolist())
            else:
                f_floors = s_floors

        # price range ===================================
        min_price = int( data['price'].min() )
        max_price = int( data['price'].max() )
        f_price = data_overview_filters.slider('Price Range', min_price, max_price, (min_price, max_price) )

        # waterfront ===================================

        wf_options = ['Waterfront', 'No Waterfront', 'Both']
        wf_select = data_overview_filters.radio('Select waterfront option:', wf_options, index=2 )

        if wf_select == 'Both':
            f_waterfront = [0, 1]
        elif wf_select == 'Waterfront':
            f_waterfront = [1]
        else:
            f_waterfront = [0]

        # Date Interval ===================================
        min_date = data['date'].min()
        max_date = data['date'].max()
        f_date = data_overview_filters.date_input('Select Date:', (min_date, max_date), min_value=min_date, max_value=max_date )
    
        # Year Built Interval =============================
        min_year_built = int(data['yr_built'].min())
        max_year_built = int(data['yr_built'].max())
        f_yr_built = data_overview_filters.slider( 'Select Year Built Range', min_year_built, max_year_built, (min_year_built, max_year_built) )

    # =======================================================================
    # filtered data transformation
    # =======================================================================

    filtered_data = data[(data['price'].between(left=f_price[0], right=f_price[1], inclusive='both')) &
                        (data['yr_built'].between(left=f_yr_built[0], right=f_yr_built[1], inclusive='both')) &
                        (data['date'].between(left=f_date[0], right=f_date[1], inclusive='both')) &
                        (data['bedrooms'].isin(f_bedrooms)) &
                        (data['bathrooms'].isin(f_bathrooms)) &
                        (data['floors'].isin(f_floors)) &
                        (data['zipcode'].isin(f_zipcodes)) &
                        (data['waterfront'].isin(f_waterfront))]


    df = filtered_data[f_attributes].reset_index().drop('index', axis=1)

    return df, filtered_data, f_zipcodes

# =======================================================================
# Data Overview
# =======================================================================
def data_overview(df, filtered_data, f_zipcodes):
    c1, c2 = st.columns((1,40))

    with c1:    # House Rocket Logo
        photo = Image.open('images/house_rocket_logo.png')
        st.image(photo, width=200)

    with c2:    # Opening Title
        st.markdown("<h2 style='text-align: center;'>Welcome to House Rocket Data Analysis</h2>", unsafe_allow_html=True)

    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Portifolio Data Overview</h3>", unsafe_allow_html=True)

    # Exhibit filtered dataset
    float_columns =df.select_dtypes( include=[ 'float64'] ).columns.tolist()
    st.dataframe(df.style.format(formatter="{:.2f}",subset=float_columns), height=200 )
    st.write(f'{df.shape[0]} properties selected.')

    c1, c2 = st.columns((1, 1))

    with c1:    # Average Values
        # Average metrics dataset ===============================================
        ids_per_zipcode = filtered_data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
        metrics_per_zipcode = filtered_data[['price', 'bedrooms', 'bathrooms', 'm2_living', 'm2_lot', 'price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()
        avg_stats = pd.merge(ids_per_zipcode, metrics_per_zipcode, on='zipcode', how='inner')
        avg_stats.columns = ['Zipcode', 'Total Houses','Average Price', 'Mean of Bedrooms', 'Mean of Bathrooms', 'Average Living Area','Average Lot Area', 'Average Price/m2']
        float_columns =avg_stats.select_dtypes( include=[ 'float64'] ).columns.tolist()
        
        df_avg = avg_stats[avg_stats['Zipcode'].isin(f_zipcodes)]
        c1.markdown("<h3 style='text-align: center;'>Average Values</h3>", unsafe_allow_html=True)
        c1.dataframe( df_avg.style.format(subset=float_columns, formatter="{:.2f}"), height=200 )

    with c2:    # Descriptive Stats
        # descriptive statistics dataset=========================================
        stats_attributes = filtered_data.select_dtypes(include=['int64', 'float64']).drop(['id', 'waterfront', 'zipcode', 'lat', 'long', 'date_year', 'date_month'], axis=1)

        max_ = pd.DataFrame(stats_attributes.apply(np.max))
        min_ = pd.DataFrame(stats_attributes.apply(np.min))
        mean_ = pd.DataFrame(stats_attributes.apply(np.mean))
        median_ = pd.DataFrame(stats_attributes.apply(np.median))
        std_ = pd.DataFrame(stats_attributes.apply(np.std))

        descriptive_stats = pd.concat([max_, min_, mean_, median_, std_], axis=1).reset_index()

        descriptive_stats.columns = ['Attributes', 'Max', 'Min', 'Mean', 'Median', 'Std. Deviation']

        desccriptive_float_columns =descriptive_stats.select_dtypes( include=[ 'float64'] ).columns.tolist()
        c2.markdown("<h3 style='text-align: center;'>Descriptive Stats</h3>", unsafe_allow_html=True)
        c2.dataframe( descriptive_stats.style.format(formatter="{:.2f}",subset=desccriptive_float_columns), height=200 )

    data_map = px.scatter_mapbox(filtered_data,
                                lat='lat',
                                lon='long',
                                color='price',
                                size='price',
                                hover_name='id',
                                hover_data=['price'],
                                color_discrete_sequence=['darkgreen'],
                                # color_continuous_scale=px.colors.cyclical.IceFire,
                                zoom=9,
                                height=300)
    data_map.update_layout(mapbox_style='open-street-map')
    data_map.update_layout(height=600, margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
    st.plotly_chart(data_map, use_container_width=True)

# =======================================================================
# Density Overview
# =======================================================================
def density_overview(df, filtered_data):
    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Geographical Overview</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns((1, 1))

    with c1:    # Density Map
        c1.markdown("<h3 style='text-align: center;'>Portifolio Density</h3>", unsafe_allow_html=True)
        df = filtered_data
        # Base Map - Folium
        density_map = folium.Map(location=[filtered_data['lat'].mean(),
                                        filtered_data['long'].mean()],
                                default_zoom_start=15)
        marker_cluster = MarkerCluster().add_to(density_map)
        for name, row in df.iterrows():
            folium.Marker([row['lat'], row['long']],
                        popup=folium.Popup('Price R${0}, since {1}, Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(row['price'],
                        row['date'],
                        row['m2_living'],
                        row['bedrooms'],
                        row['bathrooms'],
                        row['yr_built']),
                        min_width=500)).add_to(marker_cluster)
        folium_static(density_map)

    with c2:    # Price Map
        c2.markdown("<h3 style='text-align: center;'>Price Density</h3>", unsafe_allow_html=True)

        df = filtered_data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
        df.columns = ['ZIP', 'PRICE']

        geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

        region_price_map = folium.Map(location=[filtered_data['lat'].mean(),
                                                filtered_data['long'].mean()],
                                    default_zoom_start=15)

        region_price_map.choropleth(data=df,
                                    geo_data=geofile,
                                    columns=['ZIP', 'PRICE'],
                                    key_on='feature.properties.ZIP',
                                    fill_color='YlOrRd',
                                    fill_opacity=0.7,
                                    line_opacity=0.2,
                                    legend_name='AVG PRICE')
        folium_static(region_price_map)

# =======================================================================
# Temporal Analysis of Prices
# =======================================================================
def temporal_analysis(df, filtered_data):
    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Price Analysis</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns((1, 1))

    with c1:    # Prices per Year Built
        df = filtered_data[['yr_built', 'price']].groupby( 'yr_built' ).mean().reset_index()
        fig = px.line( df,
                        x='yr_built',
                        y='price',
                        title='Average price per year built' )
        c1.plotly_chart( fig, use_container_width=True )

    with c2:    # Price Distribution
        df = filtered_data
        fig = px.histogram( df,
                        x='price',
                        nbins=50,
                        title='Price Distribution' )
        c2.plotly_chart( fig, use_container_width=True )

    c1, c2, c3, c4 = st.columns((1, 1, 1, 1))

    group_att = ['bedrooms', 'bathrooms', 'floors', 'basement', 'waterfront', 'yr_built', 'date_year']
    options = data.select_dtypes(include=['int64', 'float64']).drop(['id', 'zipcode', 'lat', 'long', 'date_year', 'date_month'], axis=1).columns.to_list()
    first_att = c1.selectbox(label='Select First Attribute', options=options, index=0)
    second_att = c2.selectbox(label='Select Second Attribute', options=data[options].drop([first_att], axis=1).columns.to_list(), index=0)
    third_att = c3.selectbox(label='Select third Attribute', options=options, index=0)
    fourth_att = c4.selectbox(label='Select fourth Attribute', options=group_att, index=0)
    c1, c2 = st.columns((1,1))

    with c1:
        fig = px.scatter(data,x=first_att, y=second_att)
        c1.plotly_chart( fig, use_container_width=True )
        correlation = data[first_att].corr(data[second_att])
        c1.write(f'Correlation between {first_att} and {second_att}: {round(correlation,2)}')

    with c2:
        df = data[['price', fourth_att]].groupby(fourth_att).mean().reset_index()
        df.columns = [fourth_att, 'Mean of Prices']
        fig = px.scatter(df,x='Mean of Prices', y=fourth_att)
        c2.plotly_chart( fig, use_container_width=True )
        correlation = df['Mean of Prices'].corr(df[fourth_att])
        c2.write(f'Correlation between Mean of Prices and {fourth_att}: {round(correlation,2)}')

# =======================================================================
# Properties Attributes
# =======================================================================
def properties_attributes(df, filtered_data):
    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Properties Attributes</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns((1,1))

    with c1:    # Houses per bedrooms
        bedrooms_df = filtered_data[['bedrooms','id']].groupby('bedrooms').count().reset_index()
        bedrooms_df.columns = ['Bedrooms', 'Properties']
        bedrooms_df['Bedrooms'] = bedrooms_df['Bedrooms'].astype(str)
        rows = bedrooms_df.shape[0]
        fig = px.bar( bedrooms_df.head(rows),
                        x='Bedrooms',
                        y='Properties',
                        text_auto=True,
                        color='Properties',
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        title='Houses per Bedrooms' )
        fig.update_layout(bargap=0.2)
        c1.plotly_chart( fig, use_containder_width=True )

    with c2:    # Houses per bathrooms
        bathrooms_df = filtered_data[['bathrooms','id']].groupby('bathrooms').count().reset_index()
        bathrooms_df.columns = ['Bathrooms', 'Properties']
        bathrooms_df['Bathrooms'] = bathrooms_df['Bathrooms'].astype(str)
        rows = bathrooms_df.shape[0]
        fig = px.bar( bathrooms_df.head(rows),
                        x='Bathrooms',
                        y='Properties',
                        text_auto=True,
                        color='Properties',
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        title='Houses per Bathrooms' )
        fig.update_layout(bargap=0.2)
        c2.plotly_chart( fig, use_containder_width=True )

    c1, c2 = st.columns((1,1))

    with c1:    # Houses per floors
        floors_df = filtered_data[['floors','id']].groupby('floors').count().reset_index()
        floors_df.columns = ['Floors', 'Properties']
        floors_df['Floors'] = floors_df['Floors'].astype(str)
        rows = floors_df.shape[0]
        fig = px.bar( floors_df.head(rows),
                        x='Floors',
                        y='Properties',
                        text_auto=True,
                        color='Properties',
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        title='Houses per Floors')
        fig.update_layout(bargap=0.2)
        c1.plotly_chart( fig, use_containder_width=True )

    with c2:    # Houses per waterfront
        filtered_data['Has Waterfront'] = filtered_data['waterfront'].apply(lambda x: 'Waterfront' if x == 1 else 'No Waterfront')
        waterfront_df = filtered_data[['Has Waterfront','id']].groupby('Has Waterfront').count().reset_index()
        waterfront_df.columns = ['Waterfront', 'Properties']
        rows = waterfront_df.shape[0]
        fig = px.bar( waterfront_df.head(rows),
                        x='Waterfront',
                        y='Properties',
                        text_auto=True,
                        color='Properties',
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        title='Houses per Waterfront' )
        fig.update_layout(bargap=0.2)
        fig.update_traces(width=0.3)
        c2.plotly_chart( fig, use_containder_width=True )

# =======================================================================
# Investment Recommendations
# =======================================================================
def investment_recommendations(investment_dataset):
    st.markdown("""---""")
    st.markdown("<h3 style='text-align: center;'>Investment Recommendations</h3>", unsafe_allow_html=True)

    investment_criteria = st.expander(label='Show investment criteria')
    with investment_criteria:
        '''
        Investment criteria are as follows:  
        • Propertie price is bellow regional median.  
        • Propertie condition is above regional median.  
        • Propertie living area is above regional median.  
        • Propertie living area is above regional median.
        '''

    i1, i2 = st.columns((1,1) )

    with i1:    # Investment Overview Map
        investment_map = px.scatter_mapbox(investment_dataset,
                                            lat='lat',
                                            lon='long',
                                            color='price',
                                            size='price',
                                            hover_name='id',
                                            hover_data=['price'],
                                            color_discrete_sequence=['darkgreen'],
                                            # color_continuous_scale=px.colors.cyclical.IceFire,
                                            zoom=9,
                                            height=300)

        investment_map.update_layout(mapbox_style='open-street-map')
        investment_map.update_layout(height=600, margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
        i1.markdown("<h5 style='text-align: left;'>Recommended Properties Location</h5>", unsafe_allow_html=True)
        i1.plotly_chart(investment_map, use_container_width=True)

    with i2:    # Investment recommendation report
        i2.markdown("<h5 style='text-align: left;'>Investment Recommendation Report</h5>", unsafe_allow_html=True)
        i2.dataframe(investment_dataset, height=600)
        i2.write(f'{investment_dataset.shape[0]} properties are recommended for purchase.')

    investment = investment_dataset[['margin', 'selling_price', 'profit', 'price']].sum()[3]
    brute_return = investment_dataset[['margin', 'selling_price', 'profit', 'price']].sum()[1]
    profit = investment_dataset[['margin', 'selling_price', 'profit', 'price']].sum()[2]
    margin = investment_dataset[['margin', 'selling_price', 'profit', 'price']].mean()[0] * 100
    st.markdown(f'O investimento necessário é de ${investment:,}')
    st.markdown(f'O retorno bruto esperado é de ${brute_return:,}')
    st.markdown(f'o lucro total esperado é de ${profit:,}')
    st.markdown(f'A margem média de lucro esperada é de {margin:.2f}%')

if __name__ == '__main__':
    # =======================================================================================
    # Extraction
    # =======================================================================================

    # load dataset
    data = get_data('datasets/kc_house_data.csv')

    # =======================================================================================
    # Transformation
    # =======================================================================================

    # Treat outliers
    treat_outliers(data)

    # Correct date format
    correct_datetimes(data)

    #convert areas to metric system
    convert_to_metric(data)

    # crete new attributes
    create_new_attributes(data)
    # crete investment dataset
    create_investment_dataset(data)
    investment_dataset = create_investment_dataset(data)

    # Measures
    create_measures(data)

    #sidebar filters
    df, filtered_data, f_zipcodes = sidebar_filters(data)

    # =======================================================================================
    # Load
    # =======================================================================================

    # draw maps and dashboard
    data_overview(df, filtered_data, f_zipcodes)
    temporal_analysis(df, filtered_data)
    properties_attributes(df, filtered_data)
    properties_attributes(df, filtered_data)
    investment_recommendations(investment_dataset)

