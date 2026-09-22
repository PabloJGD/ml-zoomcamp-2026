# Homework
import pandas as pd
import numpy as np
from utils.f00_logger import logger

# Q1
logger.info("Question 1")
logger.info(f"Pandas version: {pd.__version__}")

# Q2
logger.info("Question 2")
csv_path = 'homework/data/car_fuel_efficiency_2026.csv'
df = pd.read_csv(csv_path)

logger.info(f"Dataset read! The shape is {df.shape[0]} rows and {df.shape[1]} columns")

# Q3
logger.info("Question 3")
logger.info(f"Fuel types: {df.fuel_type.unique()}")

# Q4
logger.info("Question 4")
logger.info(f"Missing Values are: {df.isnull().sum()}")

# Q5
logger.info("Question 5")
logger.info(f"Max fuel efficiency: {df.fuel_efficiency_mpg.max()}")

# Q6
logger.info("Question 6")
median_horsepower = df.horsepower.median()
mode_horsepower = df.horsepower.mode()[0]
new_median_horsepower = df.horsepower.fillna(mode_horsepower).median()

if median_horsepower == new_median_horsepower:
    logger.info(f"Median value of horsepower not changed")
elif median_horsepower > new_median_horsepower:
    logger.info(f"Median value of horsepower decreased from {median_horsepower} to {new_median_horsepower}")
else:
    logger.info(f"Median value of horsepower increased from {median_horsepower} to {new_median_horsepower}")

# Q7
logger.info("Question 7")
X = df[df.origin=='Asia'][['vehicle_weight','model_year']].head(7).to_numpy()
XT = np.transpose(X)
XTX = np.dot(XT,X)
XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = np.dot(np.dot(XTX_inv, XT),y)

logger.info(f"The sum value of w is {sum(w)}")