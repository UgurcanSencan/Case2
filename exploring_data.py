import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data_from_csv

directory_path = "C:/Users/Ugur/PycharmProjects/Case2/csv" # replace with the path to your directory
file_extension = ".csv" # replace with the file extension of your files

allergies = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[0]))
careplans = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[1]))
conditions = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[2]))
devices = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[3]))
encounters = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[4]))
imaging_studies = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[5]))
immunizations = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[6]))
medications = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[7]))
observations = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[8]))
organizations = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[9]))
patients = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[10]))
payers = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[11]))
payer_transitions = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[12]))
procedures = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[13]))
providers = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[14]))
supplies = pd.read_csv(os.path.join(directory_path, os.listdir(directory_path)[15]))

# Merge patients with medical conditions
p_with_cond = pd.merge(patients, conditions, left_on="Id", right_on="PATIENT")

# Find most common conditions for both sex and for race/ethnicity and for city
p_with_cond_filtered = p_with_cond.loc[:,['BIRTHDATE','DEATHDATE','MARITAL','RACE','ETHNICITY','GENDER','CITY','DESCRIPTION']]
# pd.to_datetime(p_with_cond_filtered.loc[:,'BIRTHDATE']).dt.year

# Find the age distribution of all patients
birthdates_count = pd.to_datetime(p_with_cond_filtered.loc[:,'BIRTHDATE']).dt.year.value_counts().sort_index()
age_average = np.mean(2023-pd.to_datetime(p_with_cond_filtered.loc[:,'BIRTHDATE']).dt.year)

# Plot a bar chart of the ten most common conditions
plt.bar(birthdates_count.index, birthdates_count)
plt.grid(True)
plt.xticks(np.arange(birthdates_count.index[0], birthdates_count.index[len(birthdates_count)-1]+1,5))
plt.title('The birth year distribution of all patients', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('The number of patient', fontsize=16)
plt.show()

# Find the gender distribution of all patients
gender_count = p_with_cond_filtered.loc[:,'GENDER'].value_counts()

# Plot a pie chart of the gender of patients
plt.pie(gender_count.to_list(), labels=gender_count.index.to_list(), startangle=90, autopct='%1.3f%%', shadow=True, explode=(0.1,0.1))
plt.title('The gender distribution of all patients', fontsize=18)
plt.axis('equal')
plt.show()

# Number of patients; died or living
p_with_cond_filtered.DEATHDATE.value_counts().sum()
p_with_cond_filtered.BIRTHDATE.value_counts().sum()

# Find the number of conditions in each city
p_eth_cond=p_with_cond_filtered.loc[:,['CITY','DESCRIPTION']]
p_eth_cond[p_eth_cond[0:len(p_eth_cond)] >= 200]
p_eth_cond_upper = p_eth_cond.value_counts()
#p_eth_cond_upper[p_eth_cond_upper>35]

# Plot a bar chart of number of conditions in each city
plt.bar(birthdates_count.index, birthdates_count)
plt.grid(True)
plt.xticks(np.arange(birthdates_count.index[0], birthdates_count.index[len(birthdates_count)-1]+1,5))
plt.title('The birth year distribution of all patients', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('The number of patient', fontsize=16)
plt.show()

# Extract data for a single patient
counts = p_with_cond["Id"].value_counts()
max_count_item = counts.idxmax()
patient_data = p_with_cond[p_with_cond["Id"] == max_count_item]

# Sort the patient data by date of encounter
p_with_cond_sorted = patient_data.sort_values('START')

# Plot a scatter plot of a patient trajectory
plt.scatter(p_with_cond_sorted['START'], p_with_cond_sorted['DESCRIPTION'])
plt.grid(True)
plt.xticks(rotation = 90)
plt.title('A Patient Trajectory', fontsize=20)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Medical Encounter', fontsize=16)
plt.show()

patient_data_demographic = patients[patients["Id"] == max_count_item]

# Find the most common conditions
common_cond_counts = conditions["DESCRIPTION"].value_counts()
count_items = common_cond_counts[common_cond_counts[0:len(common_cond_counts)] >= 200]

# Plot a bar chart of the ten most common conditions
plt.barh(count_items.index, count_items[::-1])
plt.grid(True)
plt.xticks(np.arange(150, count_items[0]+50, 50))
plt.title('Ten most common conditions', fontsize=20)
plt.ylabel('Description', fontsize=16)
plt.xlabel('Counts of each condition', fontsize=16)
plt.show()

# Similarities in how three conditions are treated
conditions_non_na = conditions.dropna()
df_similar = pd.merge(conditions_non_na, careplans.iloc[:,[3,4,5,6]], left_on=["PATIENT","ENCOUNTER"], right_on=["PATIENT","ENCOUNTER"])

countOfConditions = df_similar.groupby(['DESCRIPTION_x','DESCRIPTION_y'])
grouped_counts_for_treated_conditions = countOfConditions['DESCRIPTION_x'].count()

df_similar_sorted = df_similar.loc[:,['DESCRIPTION_x','DESCRIPTION_y']].sort_values(['DESCRIPTION_y'], ascending=True)

df = pd.DataFrame({'A': df_similar_sorted.iloc[:,0], 'B': df_similar_sorted.iloc[:,1]})
df=df.drop_duplicates()
# pivot dataframe and convert to binary matrix
binary_matrix = df.pivot(index='B', columns='A', values='B').fillna(0)

print(binary_matrix)

df1 = pd.merge_ordered(allergies.loc[:,["PATIENT","ENCOUNTER","CODE"]], conditions.loc[:,["PATIENT","ENCOUNTER","CODE"]], left_on=["PATIENT","ENCOUNTER"], right_on=["PATIENT","ENCOUNTER"])
df1 = df1.rename(columns={'CODE_x': 'CODE_allergy' , 'CODE_y' : 'CODE_cond'})

df2 = pd.merge_ordered(df1, immunizations.loc[:,["PATIENT","ENCOUNTER","CODE"]], left_on=["PATIENT","ENCOUNTER"], right_on=["PATIENT","ENCOUNTER"])
df2 = df2.rename(columns={'CODE': 'CODE_imm'})

df3 = pd.merge_ordered(df2, medications.loc[:,["PATIENT","ENCOUNTER","CODE"]], left_on=["PATIENT","ENCOUNTER"], right_on=["PATIENT","ENCOUNTER"])
df3 = df3.rename(columns={'CODE': 'CODE_medication'})

df4 = pd.merge_ordered(df3, careplans.loc[:,["PATIENT","ENCOUNTER","CODE"]], left_on=["PATIENT","ENCOUNTER"], right_on=["PATIENT","ENCOUNTER"])
df4 = df4.rename(columns={'CODE': 'CODE_care'})

df5 = pd.merge_ordered(df4, patients.loc[:,:], left_on=["PATIENT"], right_on=["Id"])
df5 = df5.iloc[:,[3,4,5,6,8,9,18,21,30]]

patient_history = df5

# find and change the nan values with zero value
patient_history.iloc[:,[0,1,2,3,4]] = patient_history.iloc[:,[0,1,2,3,4]].fillna(0)

from datetime import date
patient_history.iloc[:,5] = patient_history.iloc[:,5].fillna(date.today().strftime("%Y-%m-%d"))
patient_history.loc[:,['AGE']] = (pd.to_datetime(patient_history.iloc[:,5]) - pd.to_datetime(patient_history.iloc[:,4])).dt.days/365
patient_history.iloc[:,[6,7]] = patient_history.iloc[:,[6,7]].fillna('X')
patient_history = patient_history.drop(patient_history.iloc[:,[4,5]], axis=1)
# Find unique rows in the DataFrame
unique_df = patient_history.drop_duplicates()

import gower
distance_matrix = gower.gower_matrix(unique_df)

from sklearn.cluster import DBSCAN

# Configuring the parameters of the clustering algorithm
dbscan_cluster = DBSCAN(eps=0.05,
                        min_samples=20,
                        metric="precomputed")

# Fitting the clustering algorithm
dbscan_cluster.fit(distance_matrix)

# Adding the results to a new column in the dataframe
unique_df.loc[:,"CLUSTER"] = dbscan_cluster.labels_

np.sum(unique_df.loc[:,"CLUSTER"])
np.max(unique_df.loc[:,"CLUSTER"])

# Sort the patient data by date of encounter
unique_df_sorted = unique_df.sort_values('CLUSTER', ascending=False)

# care plan
unique_df_sorted = unique_df_sorted.rename(columns={'CODE_care': 'CODE'})
maindf1 = careplans.loc[:,["CODE","DESCRIPTION"]].drop_duplicates()
maindf1 = maindf1.sort_values('CODE',ascending=False)
df_clustered_sorted = pd.merge(unique_df_sorted, maindf1.drop_duplicates(), on=["CODE"], how='inner')
df_clustered_sorted = df_clustered_sorted.rename(columns={'CODE': 'CODE_care'})
df_clustered_sorted.loc[:,'CODE_care'] = df_clustered_sorted.loc[:,'DESCRIPTION']
# Remove the last two columns
df_clustered_sorted = df_clustered_sorted.drop(df_clustered_sorted.columns[-1:], axis=1)

# medication
df_clustered_sorted = df_clustered_sorted.rename(columns={'CODE_medication': 'CODE'})
maindf2 = medications.loc[:,["CODE","DESCRIPTION"]].drop_duplicates()
maindf2 = maindf2.sort_values('CODE',ascending=False)
df_clustered_sorted = pd.merge(df_clustered_sorted, maindf2.drop_duplicates(), on=["CODE"], how='inner')
df_clustered_sorted = df_clustered_sorted.rename(columns={'CODE': 'CODE_medication'})
df_clustered_sorted.loc[:,'CODE_medication'] = df_clustered_sorted.loc[:,'DESCRIPTION']
# Remove the last two columns
df_clustered_sorted = df_clustered_sorted.drop(df_clustered_sorted.columns[-1:], axis=1)

# immunization
df_clustered_sorted = df_clustered_sorted.rename(columns={'CODE_imm': 'CODE'})
maindf3 = immunizations.loc[:,["CODE","DESCRIPTION"]].drop_duplicates()
maindf3 = maindf3.sort_values('CODE',ascending=False)
df_clustered_sorted = pd.merge(df_clustered_sorted, maindf3.drop_duplicates(), on=["CODE"], how='inner')
df_clustered_sorted = df_clustered_sorted.rename(columns={'CODE': 'CODE_imm'})
df_clustered_sorted.loc[:,'CODE_imm'] = df_clustered_sorted.loc[:,'DESCRIPTION']
# Remove the last two columns
df_clustered_sorted = df_clustered_sorted.drop(df_clustered_sorted.columns[-1:], axis=1)

# conditions
df_clustered_sorted = df_clustered_sorted.rename(columns={'CODE_cond': 'CODE'})
maindf4 = conditions.loc[:,["CODE","DESCRIPTION"]].drop_duplicates()
maindf4 = maindf4.sort_values('CODE',ascending=False)
df_clustered_sorted = pd.merge(df_clustered_sorted, maindf4.drop_duplicates(), on=["CODE"], how='inner')
df_clustered_sorted = df_clustered_sorted.rename(columns={'CODE': 'CODE_cond'})
df_clustered_sorted.loc[:,'CODE_cond'] = df_clustered_sorted.loc[:,'DESCRIPTION']
# Remove the last two columns
df_clustered_sorted = df_clustered_sorted.drop(df_clustered_sorted.columns[-1:], axis=1)


# Sort the patient data by date of encounter
df_clustered_sorted = df_clustered_sorted.sort_values('CLUSTER',ascending=False)

import seaborn as sns
sns.pairplot(df_clustered_sorted)
sns.pairplot(df_clustered_sorted.iloc[:,0:8])


