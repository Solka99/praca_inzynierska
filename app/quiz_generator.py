import random
from kanji_alive_connection import get_kanji_info

meaning_list = ["day, sun", "month, moon", "fire", "water", "tree, wood", "gold, money", "earth, soil", "person", "child", "woman, female", "man, male", "mountain", "river", "rice field", "sky, empty", "rain", "flower", "dog", "cat", "fish", "car, vehicle", "electricity", "study, learning", "school", "ahead, previous", "life, birth", "friend", "country", "time, hour", "talk, speak", "eat, food", "drink", "see, look", "hear, listen", "read", "write", "go, carry out", "come", "exit, go out", "enter, insert", "stand, rise", "rest", "buy", "sell", "think", "know", "what", "new", "long, leader"]

onyomi_list = ["ニチ", "ジツ", "ゲツ", "ガツ", "カ", "スイ", "ボク", "モク", "キン", "コン", "ド", "ト", "ジン", "ニン", "シ", "ス", "ジョ", "ニョ", "ダン", "ナン", "サン", "セン", "デン", "クウ", "ウ", "カ", "ケン", "ビョウ", "ギョ", "シャ", "ガク", "コウ", "セン", "セイ", "ショウ", "ユウ", "コク", "ジ", "ワ", "ショク", "イン", "ケン", "ブン", "モン", "ドク", "ショ", "コウ", "ギョウ", "ライ", "シュツ", "ニュウ", "リツ", "キュウ", "バイ", "シ", "チ", "カ", "シン", "チョウ"]

kunyomi_list = ["ひ", "か", "つき", "みず", "き", "かね", "つち", "ひと", "こ", "おんな", "め", "おとこ", "やま", "かわ", "た", "そら", "あく", "あめ", "あま", "はな", "いぬ", "ねこ", "さかな", "うお", "くるま", "まなぶ", "さき", "いきる", "うまれる", "なま", "とも", "くに", "とき", "はなす", "はなし", "たべる", "くう", "のむ", "みる", "きく", "よむ", "かく", "いく", "おこなう", "くる", "でる", "だす", "いる", "はいる", "たつ", "たてる", "やすむ", "かう", "うる", "おもう", "しる", "なに", "なん", "あたらしい", "ながい"]

kanji_list = ["日", "月", "火", "水", "木", "金", "土", "人", "子", "女", "男", "山", "川", "田", "空", "雨", "花", "犬", "猫", "魚", "車", "電", "学", "校", "先", "生", "友", "国", "時", "話", "食", "飲", "見", "聞", "読", "書", "行", "来", "出", "入", "立", "休", "買", "売", "思", "知", "何", "新", "長"]



target_kanji = '安'
kanji_info_json = get_kanji_info(target_kanji)
meaning = kanji_info_json["kanji"]["meaning"]["english"]
onyomi_str = kanji_info_json["kanji"]["onyomi"]["katakana"]
onyomi = onyomi_str.split("、")[0].strip()
# onyomi = kanji_info_json["kanji"]["onyomi"]["katakana"].split(',', 1)[0]
kunyomi_str = kanji_info_json["kanji"]["kunyomi"]["hiragana"]
kunyomi = kunyomi_str.split("、")[0].strip()



def generate_meaning_quiz():
    question = f"Which kanji means '{meaning}'?"
    correct = target_kanji
    distractors = random.sample(
        [k for k in kanji_list if k != correct], 3
    )
    print("question ", question, "options ",  random.sample(distractors + [correct], 4),  "answer ", correct)
    return {
        "type": "meaning",
        "question": question,
        "options": random.sample(distractors + [correct], 4),
        "answer": correct
    }

def generate_reading_quiz():
    reading = kunyomi
    if not reading:
        return None
    question = f"What is the reading of {target_kanji}?"
    correct = reading
    distractors = random.sample(
        [k for k in kunyomi_list if k != correct], 3
    )
    print({
        "type": "reading",
        "question": question,
        "options": random.sample(distractors + [correct], 4),
        "answer": correct
    })
    return {
        "type": "reading",
        "question": question,
        "options": random.sample(distractors + [correct], 4),
        "answer": correct
    }
def generate_reading_to_kanji_quiz():
    question = f"Which of these kanji have a reading '{kunyomi}'?"
    correct = target_kanji
    distractors = random.sample(
        [k for k in kanji_list if k != correct], 3
    )
    print({
        "type": "reading_to_kanji",
        "question": question,
        "options": random.sample(distractors + [correct], 4),
        "answer": correct
    })
    return {
        "type": "reading_to_kanji",
        "question": question,
        "options": random.sample(distractors + [correct], 4),
        "answer": correct
    }



# def get_quiz(quiz_type: str, kanji_char: str):
#     KANJI_CHAR = kanji_char
#     target = get_kanji_from_db(KANJI_CHAR, kanji_entries)
#     pool = get_all_kanji_except(KANJI_CHAR, kanji_entries)
#
#     if quiz_type == "meaning":
#         return generate_meaning_quiz(target, pool)
#     elif quiz_type == "reading":
#         return generate_reading_quiz(target, pool)
#     elif quiz_type == "reading_to_kanji":
#         return generate_reading_to_kanji_quiz(target, pool)
#     else:
#         raise HTTPException(400, "Invalid quiz type")



# generate_meaning_quiz()　＃checked
# generate_reading_quiz()  #checked
# generate_reading_to_kanji_quiz() #checked
