import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_pocket_money():
    return pd.read_excel('pocket_money.xlsx')

def save_expenses(expenses_df):
    expenses_df.to_excel("pocket_money.xlsx", index=False)
    st.success("Expenses saved to pocket_money.xlsx")

def load_expenses():
    df = load_pocket_money()
    st.write("Expenses loaded from pocket_money.xlsx")
    return df


st.title("Welcome to Pocket Money!!!:bank: ")

expenses = []

date = st.date_input("Date:")
passion = st.number_input("passion:")
keep = st.number_input("keep")
amount = st.write(passion - keep)

st.markdown(
    f"""
        <style>
        .stApp {{
            background-image: url("https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjU0NmJhdGNoMy1teW50LTM0LWJhZGdld2F0ZXJjb2xvcl8xLmpwZw.jpg");
            background-attachment: fixed;
            background-size: cover;
            /* opacity: 0.3; */
        }}
        </style>
        """,
    unsafe_allow_html=True
)

if st.button("Add expense"):
    expenses.append((date, passion, keep))
    st.success("Expense added!")

if expenses:
    expenses_df = pd.DataFrame(expenses, columns=["Date", "passion", "keep"])
    st.write("Your expenses:")
    st.write(expenses_df)
    save_expenses(expenses_df)
else:
    expenses_df = load_expenses()
    if not expenses_df.empty:
        st.write("Your expenses:")
        st.write(expenses_df)
    else:
        st.write("No expenses added yet.")

if not expenses_df.empty:
    if st.button("Plot expenses"):
        fig, ax = plt.subplots()
        ax.scatter(expenses_df["passion"], expenses_df["keep"])
        plt.xlabel("passion")
        plt.ylabel("keep")
        st.pyplot(fig)

if not expenses_df.empty:
    if st.button('Train model'):

        model = LinearRegression()
        model.fit(expenses_df['keep'].values.reshape(-1, 1), expenses_df.index)

        joblib.dump(model, 'model.joblib')
        st.success("Model trained and saved to model.joblib")

if st.button('Predict keep'):
    model = joblib.load('model.joblib')

    predict_date = st.date_input("Enter the date to predict the keep:")

    reference_date = pd.to_datetime('2022-01-01')
    days_since_reference = (pd.to_datetime(predict_date) - reference_date).days

    predicted_keep = model.predict(np.array(days_since_reference).reshape(1, -1))[0]
    st.success(f"ต้องเก็บเพิ่มอีกหน่อย {predict_date}, {passion - keep}")
    
