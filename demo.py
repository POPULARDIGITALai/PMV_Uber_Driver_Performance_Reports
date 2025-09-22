import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = {
    "driver_surname": ["ANSHAD"]*7,
    "driver_email": ["anshadansha1010@gmail.com"]*7,
    "driver_phone": ["6282220841"]*7,
    "Tenure": [29]*7,
    "DP Working Plan": ["Rental"]*7,
    "total_earnings": [1437.26, 3438.26, 2368.2, 3653.34, 3259.13, 2187.33, 3296.67]
}

df = pd.DataFrame(data)

st.title("Correlation Heatmap")

# Compute correlation
corr = df.corr(numeric_only=True)

# Plot heatmap
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)

st.pyplot(fig)
