import requests

url = "https://kanjialive-api.p.rapidapi.com/api/public/kanji/"

headers = {
    "X-RapidAPI-Key": "98bb2f6e65msh411cec95c4f101bp10b84cjsn96877355acad",

    "X-RapidAPI-Host": "kanjialive-api.p.rapidapi.com"
}

def get_kanji_info(kanji):

    response = requests.get(url+kanji, headers=headers)

    if response.status_code == 200:
        data = response.json()
        # print(data)
        return data

    else:
        print("Błąd:", response.status_code)
        return 0


import streamlit as st

def show_kanji(data: dict):
    kanji = data["kanji"]
    # radical = data["radical"]
    refs = data["references"]
    examples = data["examples"]

    # Główne kolumny: znak | info | przykłady
    col_char, col_info, col_examples = st.columns([1, 1.3, 1.5])

    # ===== LEWA KOLUMNA – ZNAK + WIDEO STROKES =====
    with col_char:
        st.markdown("#### Kanji")
        st.markdown(
            f"<div style='font-size:90px; text-align:center;'>{kanji['character']}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Stroke order animation**")
        # st.video(kanji["video"]["mp4"])
        st.markdown(
            f"""
                    <video autoplay loop muted playsinline width="100%">
                        <source src="{kanji['video']['mp4']}" type="video/mp4">
                        <source src="{kanji['video']['webm']}" type="video/webm">
                        Your browser does not support the video tag.
                    </video>
                    """,
            unsafe_allow_html=True
        )
        # albo:
        # st.video(kanji["video"]["mp4"])

    # ===== ŚRODKOWA KOLUMNA – INFO O KANJI =====
    with col_info:
        st.markdown("#### Szczegóły")

        st.write(f"**Meaning:** {kanji['meaning']['english']}")
        st.write(f"**Strokes:** {kanji['strokes']['count']}")
        st.write(f"**Grade:** {refs.get('grade', '–')}")
        st.write("")
        st.markdown("---")
        st.write("**Onyomi**")
        st.write(f"{kanji['onyomi']['katakana']} ({kanji['onyomi']['romaji']})")

        st.write("**Kunyomi**")
        st.write(f"{kanji['kunyomi']['hiragana']} ({kanji['kunyomi']['romaji']})")



    # ===== PRAWA KOLUMNA – PRZYKŁADY ZDAN =====
    with col_examples:
        st.markdown("#### Examples")

        for ex in examples:
            st.markdown(f"**{ex['japanese']}**")
            st.write(ex["meaning"]["english"])
            # # opcjonalnie audio
            # st.audio(ex["audio"]["mp3"])
            # st.markdown("---")

# kanji = '火'
# get_kanji_info(kanji)

