import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, render_template, send_file,Response, request,jsonify
from pytrends.request import TrendReq
import time
import re
import os
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matrice as matrice
import tempfile
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
import plotly.graph_objs as go
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import plotly.graph_objs as go
from plotly.subplots import make_subplots



matplotlib.use('Agg')

app = Flask(__name__)
global_message = ""
global_msg_type =""
global_error =""
def get_interest_over_time(keyword, region,start_date=None, end_date=None):
    
    while True:
        try:
            if start_date and end_date:
                timeframe = f'{start_date} {end_date}'
            else:
                timeframe = 'today 5-y'  # Default to the past 5 years
            pytrend = TrendReq(hl='fr-FR', tz=360)  # Set language and timezone
            time.sleep(1)
            pytrend.build_payload(kw_list=[keyword], geo=region,timeframe=timeframe)
            data = pytrend.interest_over_time()
            if data.empty:
                print(f"No data available for '{keyword}' in region '{region}'. Skipping...")
                return None
            # Rename the column with the keyword for better visualization
            data.rename(columns={keyword: f'{keyword} Interest'}, inplace=True)
            
            return data
        except Exception as e:
            if '429' in str(e):  # Check if the error message contains '429'
                print(f"Too many requests for '{keyword}' in region '{region}'. Retrying ...")
                time.sleep(2)  # Retry after 60 seconds
            else:
                print(f"Error for '{keyword}' in region '{region}': {e}")
                return None

def save_data_to_csv(keyword, data,region):

    message=f"No data available for '{keyword}' in region '{region}'. Skipping..."
    msg_type="danger"
    error=1
    if data is not None:
        # Use the "DataCSV" folder to save the CSV files
        folder_path = "DataCSV"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        filename = os.path.join(folder_path, f"{keyword}_interest.csv")
        data.to_csv(filename)
        message=f"Data for '{keyword}' has been collected."
        #print(f"Data for '{keyword}' saved to '{filename}'.")
        print(message)
        msg_type="success"  
    return(message,msg_type,error)   

@app.route('/message', methods=['GET', 'POST'])      
def send_message():
    print("ssssssseeeendddddddd")
    global global_message
    global global_msg_type
    global global_error
    message=global_message
    msg_type=global_msg_type
    error=global_error
    print("sakakakakakakakakakak:::::::",message)
    print("sakakakakakakakakakak:::::::",msg_type)
    response_data = {
        'message': message,
        'msg_type': msg_type,
        'error': error,
        'display_message': True
    }

    return jsonify(response_data)
    
    
def test_Seasonality_Trend(keyword,data):
    if data is not None:
        # Convert the 'date' column to a datetime format
        temp_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        data.to_csv(temp_csv_file.name)
        data=pd.read_csv(temp_csv_file.name)
        data['date'] = pd.to_datetime(data['date'])
        # Group the data by month and calculate the average interest
        monthly_data = data.groupby(data['date'].dt.to_period('M')).agg({f"{keyword} Interest": "mean"}).reset_index()
        # Rename the columns
        monthly_data.columns = ['date', f"{keyword} Interest"]
        temp_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        monthly_data.to_csv(temp_csv_file.name,index=False)
        data=pd.read_csv(temp_csv_file.name)
        # Check for trend using Augmented Dickey-Fuller test
        unique_values = set(monthly_data[f"{keyword} Interest"])
        if len(unique_values) == 1:
            data['date']=pd.to_datetime(data['date'])
            # Set the date column as the index 
            data.set_index('date', inplace=True)
            print("The time series is constant and cannot be tested with ADF.")
            print(data)
            return data
        else:
            try:           
        # Perform the ADF test on the non-constant time series
                adf_test = adfuller(monthly_data[f"{keyword} Interest"])
                print("ADF Test p-value:", adf_test[1])
                # Perform Mann-Kendall test on the time series
                time_series = monthly_data[f"{keyword} Interest"]
                tau,p_value2 = kendalltau(range(len(time_series)), time_series)
                print("Mann-Kendall Test p-value for time series:", p_value2)
                # Extract the p-value from the test results
                p_value = adf_test[1]
                # significance level
                alpha = 0.05
                # Convert date column to a datetime format 
                print(data)
                data['date']=pd.to_datetime(data['date'])
                # Set the date column as the index 
                data.set_index('date', inplace=True)
                decomposition = sm.tsa.seasonal_decompose(data[f"{keyword} Interest"], model='additive')
                seasonal = decomposition.seasonal
                result = seasonal_decompose(data[f"{keyword} Interest"], model='additive', period=12)
                trend = decomposition.trend
                s=0#seasonal 
                t=0#trend
                if p_value<alpha:
                    s=1
                if p_value2<alpha:
                    t=1
                if(s==1 and t==0):
                    print("seasonality")
                    # Remove seasonality by subtracting the seasonal component from the original data
                    detrended_data=data
                    detrended_data[f"{keyword} Interest"]= data[f"{keyword} Interest"] - seasonal
                    return detrended_data
                elif(t==1 and s==0):
                    print("Trend")
                    # Extract the detrended component
                    detrended_data=data
                    #detrended_data[f"{keyword} Interest"] = data[f"{keyword} Interest"] - trend
                    #return detrended_data
                    return data
                elif(t==1 and s==1):
                    print("seasonality and trend")
                    detrended_data=data
                    detrended_data[f"{keyword} Interest"] = data[f"{keyword} Interest"] - seasonal
                    #detrended_data = detrended_data - trend
                    return detrended_data
            
                else:
                    print("no trend no seasonal")
            except Exception as e:
                #data['date']=pd.to_datetime(data['date'])
                # Set the date column as the index 
                #data.set_index('date', inplace=True)
                print("An error occurred:", str(e))
                return data
def arima_graph(folder_path):
    df=pd.read_csv(folder_path)
    df['Trimester'] = pd.to_datetime(df['Trimester'])
    date_labels=df['Trimester']
    print("aaaaaaaaaaaa:::",df['Trimester'])
    auto_arima_model = pm.auto_arima(df['Croissance'], seasonal=False, trace=True)
    p, d, q = auto_arima_model.order
    #print(f"Optimal ARIMA Order (p, d, q): ({p}, {d}, {q})")
    # Fit the ARIMA model with the optimal order
    model = ARIMA(df['Croissance'], order=(p, d, q))
    results = model.fit()
    # Define the number of trimesters to forecast
    forecast_steps = 12-4 # Adjust as needed for the number of trimesters
    # Generate forecasts for future trimesters
    forecast = results.forecast(steps=forecast_steps)
    # Calculate 95% confidence intervals
    residuals = results.resid
    stderr = np.std(residuals)
    # Fit the ARIMAX model with the optimal order
    df2=pd.read_csv('PCACSV/filtered_newpca.csv')
    model_arimax = ARIMA(df['Croissance'], order=(p, d, q), exog=df2['PC1'])
    results_arimax = model_arimax.fit()
    residuals = results_arimax.resid
    stderr_arimax = np.std(residuals)
    # Generate forecasts for future trimesters using ARIMAX
    forecast_arimax = results_arimax.forecast(steps=forecast_steps,exog=df2['PC1'].tail(8))
    lower_bound_arimax = forecast_arimax - 1.96 * stderr_arimax  # 95% confidence interval lower bound
    upper_bound_arimax = forecast_arimax + 1.96 * stderr_arimax
    # Calculate the upper and lower bounds of the confidence interval
    lower_bound = forecast - 1.96 * stderr  # 95% confidence interval lower bound
    upper_bound = forecast + 1.96 * stderr 
    df['Trimester'] = pd.to_datetime(df['Trimester'])
    df.set_index('Trimester', inplace=True)
    last_trimester = df.index[-1]
    print(last_trimester)
    date= last_trimester
    print("daaaaaaaaaate: ",date)
    forecast_index = pd.date_range(start=date, periods=forecast_steps+1, freq='QS-JAN')
    # Print the forecast values and dates
    print("Forecast Values:")
    print(forecast)
    print("\nForecast Dates:")
    print(forecast_index)
    #date_final=pd.concat([df.index,forecast_index[:1]])
    forecast_index=pd.to_datetime(forecast_index)  
    date_labels2=pd.to_datetime(forecast_index) 
    forecast_df = pd.DataFrame({'Date': forecast_index[1:], 'Value': forecast})
    # Save the forecast data as a CSV file
    forecast_df.to_csv("PCACSV/forecast.csv", index=False)
    date_labels2 = forecast_df['Date']
    date_labels2 = pd.to_datetime(date_labels2)
    date_labels = pd.concat([date_labels, date_labels2])
    date_labels = date_labels.dt.strftime('%b %d, %Y')
    # Create a figure
    formatted_forecast_dates = forecast_df['Date'].dt.strftime('%b %d, %Y')
    # Add forecast dates to the date_labels
    date_labels = date_labels.tolist() + formatted_forecast_dates.tolist()
    fig2 = go.Figure()
    # Plot the original data
    # Plot the original data
    fig2.add_trace(go.Scatter(
        x=df.index,
        y=df['Croissance'],
        mode='lines+markers',
        name='Original Data',
        marker=dict(symbol='circle', size=6),
        line=dict(color='blue', dash='solid')
    ))

    # Plot the forecast data with confidence intervals
    fig2.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Value'],
        mode='lines+markers',
        name='Forecast',
        marker=dict(symbol='circle', size=6, color='red'),
        line=dict(color='red', dash='dash')
    ))

    # Add 95% confidence intervals
    fig2.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=lower_bound,
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='95% Confidence Interval'
    ))

    fig2.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=upper_bound,
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='95% Confidence Interval',
        fill='tonexty'  # Fill area between lower and upper CI
    ))
    fig2.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_arimax,
        mode='lines+markers',
        name='Forecast ARIMAX',
        marker=dict(symbol='circle', size=6, color='green'),
        line=dict(color='green', dash='dash')
    ))


    fig2.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=lower_bound_arimax,
        mode='lines',
        line=dict(color='rgba(0, 128, 0, 0.002)', dash='dash'),
        name='ARIMAX 95% Confidence Interval'
    ))

    fig2.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=upper_bound_arimax,
        mode='lines',
        line=dict(color='rgba(0, 128, 0, 0.002)', dash='dash'),
        name='ARIMAX 95% Confidence Interval',
        fill='tonexty'  # Fill area between lower and upper CI
    ))
        # Add forecast dates to the date_labels
        #date_labels2 = date_labels + forecast_df['Date'].dt.strftime('%b %d, %Y')

        # Update the layout with the combined date_labels
    fig2.update_xaxes(
            tickvals=list(df.index) + list(forecast_df['Date']),
            ticktext=date_labels,
            tickangle=45,
            title='Trimester'
    )

        # Update the layout
    fig2.update_layout(
            title='ARIMA Forecast',
            yaxis_title='Croissance',
            xaxis_title='Trimester',
            legend=dict(x=0.01, y=0.99),
            showlegend=True,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            hovermode='x',
    )

    fig2.update_layout(height=400, width=1200)

    # Convert the figure to JSON
    plot_json = fig2.to_json()

    # Return the JSON representation of the figure
    return plot_json
def arima_plotly_graph(folder_path):
    df = pd.read_csv(folder_path)
    df['Trimester'] = pd.to_datetime(df['Trimester'])
    date_labels = df['Trimester']

    auto_arima_model = pm.auto_arima(df['Croissance'], seasonal=False, trace=True)
    p, d, q = auto_arima_model.order

    model = ARIMA(df['Croissance'], order=(p, d, q))
    results = model.fit()

    forecast_steps = 12 - 4
    forecast = results.forecast(steps=forecast_steps)

    last_trimester = df['Trimester'].iloc[-1]
    date = last_trimester + pd.DateOffset(months=3)
    forecast_index = pd.date_range(start=date, periods=forecast_steps + 1, freq='Q-JAN')

    # Create a new Plotly figure
    fig = make_subplots(rows=1, cols=1)

    # Plot the original data
    fig.add_trace(go.Scatter(x=df['Trimester'], y=df['Croissance'], mode='lines+markers', name='Original Data'))

    # Plot the forecast
    fig.add_trace(go.Scatter(x=forecast_index[1:], y=forecast, mode='lines+markers', name='Forecast', line=dict(color='red')))

    # Update layout for better visualization
    fig.update_layout(
        title='ARIMA Forecast',
        xaxis_title='Trimester',
        yaxis_title='Croissance',
        xaxis=dict(
            tickmode='array',
            tickvals=np.linspace(0, len(date_labels) - 1, 10, dtype=int),
            ticktext=date_labels.dt.strftime('%b %Y')[np.linspace(0, len(date_labels) - 1, 10, dtype=int)],
            tickangle=45,
        ),
    )

    # Optionally, you can save the interactive Plotly plot as an HTML file
    #fig.write_html('arima_forecast.html')

    # You can also display the plot in a Jupyter Notebook using the following line:
    fig.show()
def read_combine_csv_files(folder_path):
    combined_df = pd.DataFrame()
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)
            
            # Assuming the CSV files have a "date" and "<keyword> Interest" column
            combined_df = pd.concat([combined_df, df])
    
    return combined_df
def clear_data_csv_folder(folder):
    folder_path = folder
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"File '{file_path}' removed.")
        print(f"Contents of '{folder_path}' cleared.")

def perform_pca_and_plot(matrix):
    img_PCA=None
    # Convert matrix to a DataFrame
    df = pd.DataFrame(matrix)
    # Extract header and clear data CSV folder
    header = matrix[0][1:]
    clear_data_csv_folder("PCACSV")
    # Define CSV file paths
    csv_file_path = 'PCACSV/pca.csv'
    # Drop the first column and set new column headers
    df = df.drop(df.columns[0], axis=1)
    df = df.iloc[1:]
    df.columns = header
    # Save the DataFrame to a CSV file
    #df.to_csv(csv_file_path, index=False)
    # Read the CSV file and standardize the data
    #df = pd.read_csv(csv_file_path)
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df)
    # Perform PCA
    pca = PCA()
    pca.fit(df_standardized)
    # Transform the data to the first two principal components
    components = 1
    df_pca = pca.transform(df_standardized)[:, :components]
    # Create a new DataFrame with the PCA results
    df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(components)])
    # Save the PCA results to a CSV file
    df_pca.to_csv('PCACSV/pca_results.csv', index=False)
    # Plot the PCA results
    img_PCA = Plot_Graph('PCACSV/pca_results.csv')
    #img_PCA=None
    return img_PCA
def matrix_CSV():
    print("iiiiiiiiiimmmmheeeeeeereee")
    matrix = []
    matrix = matrice.initializeMatrix_Date("DataCSV", matrix)
    matrix = matrice.read_csv_files("DataCSV", matrix)
    df = pd.DataFrame(matrix)
    clear_data_csv_folder('MatrixCSV')
    print(matrix[0])
    df.to_csv('MatrixCSV/matrix.csv', index=False)
   # matrice.export_matrix_to_csv(matrix, 'MatrixCSV/matrix.csv')
    return matrix

@app.route('/download-csv', methods=['POST'])
def downloadCSV():
    
    matrix = []
    matrix = matrice.initializeMatrix_Date("DataCSV", matrix)
    matrix = matrice.read_csv_files("DataCSV", matrix)
    print(matrix)
    df = pd.DataFrame(matrix)
    df.to_csv('MatrixCSV/matrix.csv', index=False)
    temp_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    matrice.export_matrix_to_csv(matrix, temp_csv_file.name)
    
    #df=pd.read_csv("MatrixCSV/matrix.csv")
    #print(df)
    return send_file(temp_csv_file.name, as_attachment=True, download_name="matrix.csv")
def get_trimester(date):
    year=date.year
    if date.month in [1, 2, 3]:
        return f"{year}Q1"
    elif date.month in [4, 5, 6]:
        return f"{year}Q2"
    elif date.month in [7, 8, 9]:
        return f"{year}Q3"
    else:
        return f"{year}Q4"
def Plot_Graph(path):
    csv_file_path = path
    df = pd.read_csv(csv_file_path)#to add values of pca to the new trimister pca(pca_results)
    df2 = pd.read_csv('MatrixCSV/matrix.csv',skiprows=1)#to extract only dates
    df3 = pd.read_csv(csv_file_path)
    #df3 = pd.read_csv('MatrixCSV/matrix.csv',skiprows=1)
    # Extract the 'date' and 'PCA' col`umns
    #df2 = df2.columns[0]
    date = df2['Date']
    date = pd.to_datetime(date)
    date2=date
    date2.to_csv('PCACSV/pcaTrimseral.csv',index=False)
    df3=pd.read_csv('PCACSV/pcaTrimseral.csv')
    df3['PC1']=df['PC1']
    df3['Trimester']= date2.apply(get_trimester)
    ####################
    unique_trimesters = df3['Trimester'].unique()
    # Create an empty DataFrame to store the results
    #result_df = pd.DataFrame(columns=['Date', 'PC1', 'Trimester'])
    dates = []
    avg_pc1_values = []
    trimesters = []
    for trimester in unique_trimesters:
        # Filter the DataFrame for the current trimester
        trimester_data = df3[df3['Trimester'] == trimester]#kanjm3 ga3 douk trimister li fhal fhal 
        
        # Calculate the average PC1 for the current trimester
        avg_pc1 = trimester_data['PC1'].mean()
        
        # Get the date from the first row of the current trimester
        date3 = trimester_data['Date'].iloc[0]
        
        # Append the result to the result DataFrame
        dates.append(date3)
        avg_pc1_values.append(avg_pc1)
        trimesters.append(trimester)
    ####################
    result_df = pd.DataFrame({'Trimester':  pd.to_datetime(trimesters),'PC1': avg_pc1_values})
    
    result_df.to_csv('PCACSV/new_pcaTrimseral.csv',index=False)
    df3.to_csv('PCACSV/pcaTrimseral.csv',index=False)
    df4= pd.read_csv('ECOGRAPHCSV/Croissance_Ecoo.csv')
    df4['Trimester']=pd.to_datetime(df4['Trimester'])
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    filter_synchronize_dates(result_df,df4)
    pca_croissance=pd.read_csv('PCACSV/filtered_Croissance.csv')
    date_trimester=pca_croissance['Trimester']
    date_trimester=pd.to_datetime(date_trimester)
    pca_trimester=pd.read_csv('PCACSV/filtered_newpca.csv')
    #date4=pd.to_datetime(date4)
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    date_labels = date_trimester.dt.strftime('%b %d, %Y')
    # Create traces for PCA 1 and PCA 2
    trace_pca1 = go.Scatter(
        x=date_trimester,
        y=pca_trimester['PC1'],
        mode='lines+markers',
        name='PCA 1',
        line=dict(color='blue', dash='solid'),
        marker=dict(symbol='circle', size=8, color='blue')
    )

    trace_pca2 = go.Scatter(
        x=date_trimester,
        y=pca_croissance['Croissance'],
        mode='lines+markers',
        name='PCA 2',
        line=dict(color='red', dash='dash'),
        marker=dict(symbol='x', size=8, color='red')
    )
    # Create a layout
    layout = go.Layout(
    xaxis=dict(
        title='Quarter',
        tickvals=date_trimester,
        ticktext=date_labels,
        tickangle=45,
        gridcolor='lightgray',  # Specify grid color for x-axis
    ),
    yaxis=dict(title='PCA Value', gridcolor='lightgray'),  # Specify grid color for y-axis
    title='PCA Graph',
    legend=dict(
        x=0.01,
        y=0.99,
        traceorder="normal",
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="rgba(0, 0, 0, 0.5)",
        borderwidth=1,
    )
)

    # Create the figure
    fig2 = go.Figure(data=[trace_pca1, trace_pca2], layout=layout)
    fig2.update_layout(height=400, width=1200)
    plot_json = fig2.to_json()   
    return plot_json

def filter_synchronize_dates(file1,file2):
    file2['Trimester']=pd.to_datetime(file2['Trimester'])
    end_date_PCA_Trimisteral=file1['Trimester'].iloc[-1]#end date dyal lkeyword li 9lbna 3liha 
    end_date_PCA_Croissance=file2['Trimester'].iloc[-1]#end date dyal lfichier de coroissance economique 
    if(end_date_PCA_Trimisteral<end_date_PCA_Croissance):
        end_date_PCA=end_date_PCA_Trimisteral
    else: end_date_PCA=end_date_PCA_Croissance
    print("keyword date:",end_date_PCA_Trimisteral)
    print("Croissance date :",end_date_PCA_Croissance)
    print("end date:",end_date_PCA)
    file2.set_index('Trimester', inplace=True)
    start_date_PCA=file1['Trimester']
    file1.set_index('Trimester', inplace=True)
    #start_date_PCA=pd.to_datetime(start_date_PCA)
    start_date_PCA=start_date_PCA.iloc[0]

    #print(":",start_date_PCA)
    #df4['Trimester'] = pd.to_datetime(df4['Trimester'])
    filtered_Croissance = file2[file2.index >= start_date_PCA]
    filtered_Croissance = filtered_Croissance[filtered_Croissance.index <= end_date_PCA]
    filtered_newpca=file1[file1.index <= end_date_PCA]#do an end date (same end date for Croissance_ecoo.csv) for pcaTrimseral.csv 
    filtered_Croissance=filtered_Croissance['Croissance']
    #filter_newpca=filtered_newpca['']
    #filtered_date=filtered_newpca[start_date_PCA:]
    start_PCA=filtered_newpca['PC1']
    filtered_newpca['PC1'].to_csv('PCACSV/filtered_newpca.csv')
    filtered_Croissance.to_csv('PCACSV/filtered_Croissance.csv')
@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    img_PCA = None
    region=None
    matrix=[]
    plotly_PCA=None
    plot_json=None
    plotly_arima=None
    random_keywords = []
    error=None
    error_message=None
    if request.method == 'POST':
        
        print("imhere")
        keywords = request.form.get('keywords')
        uploaded_file = request.files.get('file_input')
        region = request.form.get('region')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        if keywords:
            keywords = [re.sub(r'[^a-zA-Z]', '', kw.strip()) for kw in keywords.split(',') if kw.strip()]
            random_keywords = keywords
        elif uploaded_file:
            keywords = uploaded_file.read().decode('utf-8').split(',')  
            keywords = [re.sub(r'[^a-zA-Z]', '', kw.strip()) for kw in keywords if kw.strip()]
            cleaned_keywords = keywords
    
            #print(keywords)   
            #random_keywords = keywords  

            random_keywords = cleaned_keywords
        else:
            random_keywords = []

        if random_keywords:
            
            clear_data_csv_folder("DataCSV")
            for keyword in random_keywords:
                data =  get_interest_over_time(keyword, region=region, start_date=start_date, end_date=end_date)
                #save_data_to_csv(keyword, test_Seasonality_Trend(keyword,data),region)
                message,msg_type,error=save_data_to_csv(keyword, test_Seasonality_Trend(keyword,data),region)
                global global_message 
                global global_msg_type
                global global_error
                global_message=message
                global_msg_type=msg_type
                global_error=error
                send_message()
                #save_data_to_csv(keyword, data)
                #return render_template('index33.html',message=message,msg_type=msg_type,error=error)
            time.sleep(1)
            global_message=None
            global_msg_type=None
            folder_path = "DataCSV" 

            if not os.listdir(folder_path):
                print(f"The folder '{folder_path}' is empty.")
                #return render_template('index33.html')
                #return render_template('index33.html',error_message='Problem with the server data, please retry...')
            else:
                combined_data = read_combine_csv_files(folder_path)
                #combined_data = combined_data.drop(columns='isPartial')
                combined_data['date'] = pd.to_datetime(combined_data['date'])

                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(1, 1, 1)   
                keywords = combined_data.columns[1:]
                fig2 = go.Figure()
                for keyword in keywords:
                    fig2.add_trace(go.Scatter(x=combined_data['date'], y=combined_data[keyword], name=keyword))
                fig2.update_xaxes(title_text='Date')
                fig2.update_yaxes(title_text='Interest')
                fig2.update_layout(title=f'Global Graph for Keywords Interests in Region: {region}')
                fig2.update_layout(legend=dict(x=0, y=1))
                fig2.update_layout(showlegend=True)
                fig2.update_layout(plot_bgcolor="rgba(255, 255, 255, 0.7)")
                fig2.update_layout(height=400, width=1200)
                plot_json = fig2.to_json()
                canvas = FigureCanvas(fig)
                img_stream = BytesIO()
                canvas.print_png(img_stream)
                img_stream.seek(0)
                #plt.clf()
                #plt.close()
                
                matrix=matrix_CSV()
                try:
                    plotly_PCA=perform_pca_and_plot(matrix)
                except Exception as e:
                    error_message="Error: The provided data does not contain enough observations for ploting graphs."
                    print("An error occurred during PCA:")
                    return render_template('index33.html',error_message=error_message)
                try:
                    plotly_arima=arima_graph("PCACSV/filtered_Croissance.csv")
                except Exception as e:
                    error_message="Error: The provided data does not contain enough observations for ploting graphs."
                    print("An error occurred during PCA:")
                    return render_template('index33.html',error_message=error_message)

    return render_template('index33.html', csv_download_link="/download-csv", plotly_PCA=plotly_PCA, region=region,keywords=random_keywords, graph_json=plot_json,plotly_arima=plotly_arima)

if __name__ == '__main__':
    app.run(debug=True)