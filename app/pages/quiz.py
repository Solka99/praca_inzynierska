import streamlit as st
from app_helpers.quiz_generator import generate_random_quiz
# wyświetlanie jsona bez formatowania
# st.title("Quizy")
# st.write("Hello,", st.session_state.get("name", "anonymous"), "👋")
#
# st.write(generate_random_quiz())

#z formatowaniem
st.title("Quiz")

quiz = generate_random_quiz()

question = quiz.get("question", "Brak pytania")
options = quiz.get("options", [])

# Karta quizowa
st.markdown("### Question:")
st.markdown(f"**{question}**")

st.write("")  # odstęp

# Wyświetlanie odpowiedzi jako ładnych przycisków
for i, option in enumerate(options):
    st.button(f"{i}. {option}", key=f"option_{i}")
