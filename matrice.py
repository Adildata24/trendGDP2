import pandas as pd
from flask import Flask, render_template, send_file,Response, request
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import shutil
import csv
import numpy as np
matrix = []
row=[]
p=0
def save_data_to_csv(keyword, data):
    if data is not None:
        # Use the "DataCSV" folder to save the CSV files
        folder_path = "DataCSV"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = os.path.join(folder_path, f"{keyword}_interest.csv")
        data.to_csv(filename)
        print(f"Data for '{keyword}' saved to '{filename}'.")
def read_csv_files(folder_path,matrix):
    column_headers = ['Date']
    num_rows = numberoflines("DataCSV")
    print("number::::",num_rows)
    num_columns = numberofwords("DataCSV")+1
    q=0
    fill_column=0
    for filename in os.listdir(folder_path):
        q=0
        col=1
        fill_column=fill_column+1
        if filename.endswith('.csv'):
            column_headers.append(filename[:-4])#to not add the extension 
            print(column_headers)
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)
            for q in range(num_rows-1):
                row = []
                first_date = df.iloc[:, 1][q]
                #print(first_date)
                fill_value = first_date
                for col in range(num_columns):
                   # print("[ ",q, "] [ ",col,"] = ")
                    if col == fill_column:
                        matrix[q][col]=fill_value
                        row.append(matrix[q][col])    
                    else:
                         row.append(matrix[q][col])  
    matrix_with_headers = np.vstack((column_headers, matrix))  
    return matrix_with_headers

def numberofwords(folder):
    folder_path = folder  # Replace with the actual path to your folder

# List all files in the folder
    files = os.listdir(folder_path)

# Count the number of files (excluding directories)
    num_files = len([f for f in files if os.path.isfile(os.path.join(folder_path, f))])
    return num_files

def numberoflines(folder):
    filename =os.listdir(folder)[0]
    filepath = os.path.join(folder, filename)
    with open(filepath, 'r') as file:
         lines = file.readlines()
    return len(lines)
def initializeMatrix_Date(folder,matrix):
    filename =os.listdir(folder)[0]
    if not filename:
        return None
    filepath = os.path.join(folder, filename)
    num_rows = numberoflines("DataCSV")
    num_columns = numberofwords("DataCSV")+1
    fill_column = 0  # Index of the column to fill
    df = pd.read_csv(filepath)
    q=0
    for _ in range(num_rows-1):
        row = []
        first_date = df['date'][q]
        q=q+1
        fill_value = first_date
        for col in range(num_columns):
            if col == fill_column:
                row.append(fill_value)
            else:
                row.append(None)
        matrix.append(row)
        
    return matrix
#read_csv_files("DataCSV")
def export_matrix_to_csv(matrix, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in matrix:
            csvwriter.writerow(row)
#numberofwords("DataCSV")
#print(numberofwords("DataCSV"))
#print(numberoflines("DataCSV"))
#matrix=initializeMatrix_Date("DataCSV",matrix)
#matrix=read_csv_files("DataCSV",matrix)
#export_matrix_to_csv(matrix,"DataCSV/matrix.CSV")
#for row in matrix:
    #print(row)