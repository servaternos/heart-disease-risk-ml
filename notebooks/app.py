import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Оценка сердечно-сосудистого риска")
st.write("Введите клинические показатели пациента")

df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Возраст", min_value=29, max_value=77, value=54)
    sex = st.selectbox("Пол", ["Женский", "Мужской"])
    cp = st.selectbox("Тип боли в груди", [0, 1, 2, 3])
    trestbps = st.number_input("Давление в покое (мм рт.ст.)", min_value=90, max_value=200, value=130)
    chol = st.number_input("Холестерин (мг/дл)", min_value=120, max_value=570, value=240)
    fbs = st.selectbox("Сахар > 120 мг/дл", ["Нет", "Да"])
    restecg = st.selectbox("ЭКГ в покое", [0, 1, 2])

with col2:
    thalach = st.number_input("Макс. пульс при нагрузке", min_value=70, max_value=202, value=150)
    exang = st.selectbox("Стенокардия при нагрузке", ["Нет", "Да"])
    oldpeak = st.number_input("Депрессия ST", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope = st.selectbox("Наклон ST", [0, 1, 2])
    ca = st.number_input("Количество сосудов (0-4)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Талассемия", [0, 1, 2, 3])

sex_val = 1 if sex == "Мужской" else 0
fbs_val = 1 if fbs == "Да" else 0
exang_val = 1 if exang == "Да" else 0

if st.button("Рассчитать риск"):
    input_data = np.array([
        [age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")

    if prediction == 1:
        st.error(f"Риск ВЫСОКИЙ (вероятность: {probability[1] * 100:.1f}%)")
    else:
        st.success(f"Риск НИЗКИЙ (вероятность: {probability[0] * 100:.1f}%)")

    st.markdown("---")
    st.subheader("Вклад признаков в решение модели")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Признак": X.columns, "Важность": importances})
    fi_df = fi_df.sort_values("Важность", ascending=True)
    st.bar_chart(fi_df.set_index("Признак"))