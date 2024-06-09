import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from time import sleep


def collect_data_for_item(object_name, duration, file_path='video.csv', select_columns=None):
    input("Press Enter to start the timer and keep looking at the item...")

    
    time.sleep(duration)
    df = pd.read_csv(file_path, header=0)
    df.columns = df.columns.str.strip()  
    data = df[select_columns].copy()
    data['State'] = object_name

    return data


select_columns = ['pose_Rx', 'pose_Ry', 'pose_Rz', 'pose_Tx', 'pose_Ty', 'pose_Tz']


combined_collected_data = pd.DataFrame()


def clear_video_csv(file_path='video.csv'):
    df = pd.read_csv(file_path, header=0)
    df = df.head(2)
    df.to_csv(file_path, index = False)


o1 = input("Enter name of item 1: ")
collected_data_o1 = collect_data_for_item(o1, 10, select_columns=select_columns)
combined_collected_data = pd.concat([combined_collected_data, collected_data_o1], ignore_index=True)


clear_video_csv(file_path='video.csv')


o2 = input("Enter name of item 2: ")
collected_data_o2 = collect_data_for_item(o2, 10, select_columns=select_columns)
combined_collected_data = pd.concat([combined_collected_data, collected_data_o2], ignore_index=True)


clear_video_csv(file_path='video.csv')

o3 = input("Enter name of item 3: ")
collected_data_o3 = collect_data_for_item(o3, 10, select_columns=select_columns)
combined_collected_data = pd.concat([combined_collected_data, collected_data_o3], ignore_index=True)

combined_collected_data.to_csv('collected_data.csv', index=False)

print("Collected data saved to 'collected_data.csv'.")


y = combined_collected_data['State']
X = combined_collected_data[['pose_Rx', 'pose_Ry', 'pose_Rz', 'pose_Tx', 'pose_Ty', 'pose_Tz']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


lr = LogisticRegression(multi_class='ovr', max_iter=1000)
lr.fit(X_train, y_train)



while True:
    df_updated = pd.read_csv('video.csv', header=0)
    df_updated.columns = df_updated.columns.str.strip()
    last_row = df_updated.iloc[-1:]
    new_data = last_row[select_columns]

    new_y_pred = lr.predict(new_data)
    print("New predictions:", new_y_pred)

    sleep(0.5)




