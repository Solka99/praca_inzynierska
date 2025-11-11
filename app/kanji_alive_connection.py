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

# kanji = '火'
# get_kanji_info(kanji)

