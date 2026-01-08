import requests

url = "https://kanjialive-api.p.rapidapi.com/api/public/kanji/"

headers = {
    "X-RapidAPI-Key": "", #do uzupełnienia

    "X-RapidAPI-Host": "" #do uzupełnienia
}

def get_kanji_info(kanji):
    hiragana_list = [
        'あ', 'い', 'う', 'え', 'お',
        'か', 'き', 'く', 'け', 'こ',
        'さ', 'し', 'す', 'せ', 'そ',
        'た', 'ち', 'つ', 'て', 'と',
        'な', 'に', 'ぬ', 'ね', 'の',
        'は', 'ひ', 'ふ', 'へ', 'ほ',
        'ま', 'mi', 'む', 'め', 'も',
        'や', 'ゆ', 'よ',
        'ら', 'り', 'る', 'れ', 'ろ',
        'わ', 'を', 'ん',
        'ぁ', 'ぃ', 'ぅ', 'ぇ', 'ぉ',
        'っ', 'ゃ', 'ゅ', 'ょ', 'ゎ'
    ]
    if kanji in hiragana_list:
        return 0
    response = requests.get(url+kanji, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(data)
        return data

    else:
        print("Błąd:", response.status_code)
        return 0


import streamlit as st

def show_kanji(data: dict):
    if not data:
        st.write("Nie ma dodatkowych informacji o znakach hiragana.")
        return 0
    print(type(data))
    print(data.keys())

    kanji = data['kanji']
    examples = data["examples"]

    col_char, col_info, col_examples = st.columns([1, 1.3, 1.5])

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


    with col_info:
        st.markdown("#### Szczegóły")

        st.write(f"**Meaning:** {kanji['meaning']['english']}")
        st.write(f"**Strokes:** {kanji['strokes']['count']}")
        st.write("")
        st.markdown("---")
        st.write("**Onyomi**")
        st.write(f"{kanji['onyomi']['katakana']} ({kanji['onyomi']['romaji']})")

        st.write("**Kunyomi**")
        st.write(f"{kanji['kunyomi']['hiragana']} ({kanji['kunyomi']['romaji']})")



    with col_examples:
        st.markdown("#### Examples")

        for ex in examples[:5]:
            st.markdown(f"**{ex['japanese']}**")
            st.write(ex["meaning"]["english"])


# kanji = '火'
# get_kanji_info(kanji)

