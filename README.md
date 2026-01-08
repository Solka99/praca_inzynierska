# Aplikacja do rozpoznawania ręcznie pisanych znaków kanji (CNN + Streamlit)

Projekt powstał w ramach pracy inżynierskiej, której celem było stworzenie modelu uczenia maszynowego pozwalającego na rozpoznawanie ręcznie pisanych japońskich znaków.
Aplikacja umożliwia:
- rysowanie znaków przez użytkownika (myszka/rysik),
- klasyfikację znaków z użyciem wytrenowanego modelu CNN,
- ocenę jakości rysunku (wynik 0–100),
- prezentację informacji o rozpoznanym znaku.

## Struktura projektu (wysoki poziom)

Aplikacja została podzielona na moduły:
1. **Moduł rysowania znaków** – interaktywny canvas.
2. **Moduł klasyfikacji znaków** – predykcja klasy kanji przy użyciu modelu CNN.
3. **Moduł oceny jakości rysunku** – obliczanie miar podobieństwa i wynik 0–100.
4. **Moduł informacji o znaku** – pobieranie danych o kanji z KanjiAlive API.

## Wymagania

- Python 3.10+
- pip

## Instalacja i uruchomienie

### 1) Klonowanie repozytorium
```bash
git clone <URL_REPO>
cd <NAZWA_FOLDERU>
```
Aby uruchomić aplikację należy wpisać w terminalu:
```bash 
streamlit run app/main.py 
```
Aby projekt zadział poprawnie należy:
1. Pobrać zbiór danych ETL8B z http://etlcdb.db.aist.go.jp/download2/ i zapisać go do folderu **data**. Zbiór nie został dołączony do repozytorium ze względu na rozmiar.
2. Uzupełnić X-RapidAPI-Key oraz X-RapidAPI-Host w pliku kanji_allive_connection, aby uzyskać połączenie z Kanji Alive API. Dane te można wygenerować na stronie: https://kanjialive-api.p.rapidapi.com/api/public/kanji/. (dostęp 01.2026)
