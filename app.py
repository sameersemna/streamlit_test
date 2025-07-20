import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import re
import io # To capture output of .info()

# You might not need all these imports for the current code, but keeping them as per original
# from concurrent.futures import ThreadPoolExecutor
# from gensim.models import Word2Vec
# from imblearn.over_sampling import SMOTE
# from PIL import Image
# from scipy.stats import entropy, kstest, linregress
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.metrics import mean_squared_error, f1_score, classification_report, confusion_matrix, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.svm import LinearSVC
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
# from tensorflow.keras.metrics import F1Score
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import AdamW
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.regularizers import l2
# import ast
# import base64
# import cv2
# import imagehash
# import time
# import warnings
# import xgboost as xgb

# --- PAGE CONFIG MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="Rakuten E-commerce Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading (Cached for efficiency) ---
@st.cache_data
def load_all_data():
    path = r"./data/raw"
    try:
        X_test_df = pd.read_parquet(f"{path}/X_test_update.parquet")
        X_train_df = pd.read_parquet(f"{path}/X_train_update.parquet")
        Y_train_df = pd.read_parquet(f"{path}/Y_train_CVw08PX.parquet")
        # Merge Y_train into X_train immediately after loading to ensure consistency across reruns
        X_train_df = X_train_df.merge(Y_train_df, how='left', left_index=True,
                                    right_index=True, suffixes=('_X_train', '_Y_train'))
        return X_test_df, X_train_df, Y_train_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please ensure data files are in the '{path}' directory.")
        st.stop() # Stop the app if data is not found
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        st.stop()

X_test, X_train, Y_train = load_all_data()

# --- NLTK Downloads (Cached for efficiency) ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt') # Required for word_tokenize, though not explicitly used here, good to have if needed later
    except nltk.downloader.DownloadError:
        nltk.download('punkt')

download_nltk_data()


st.title("Rakuten e-commerce project")
st.sidebar.title("Table of contents")
pages = ["Data Processing", "DataVizualization", "Modelling"]
page = st.sidebar.radio("Go to", pages)


# --- Page 1: Data Processing ---
if page == pages[0]:
    st.header("Display the original train data")

    st.subheader("Presentation of DataFrames")

    tables = [('X_train', X_train), ('X_test', X_test), ('Y_train', Y_train)]
    for table_name, table in tables:
        st.markdown(f"\n**{table_name}**\n")
        st.dataframe(table.head())

        # Capture .info() output
        buffer = io.StringIO()
        table.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.write(f"Duplicates: {table.duplicated().sum()}")

    st.subheader("The training data (merged with Y_train)")
    st.dataframe(X_train.head())
    st.write(f"Shape of X_train: {X_train.shape}")

    st.subheader("The dataset description")
    st.dataframe(X_train.describe())

    if st.checkbox("Show NA counts for X_train"):
        st.dataframe(X_train.isna().sum())


    st.markdown("""
    "In each column, we are going to investigate:
    1. Missing values
    2. Duplicates
    3. Unique modalities"
    """)

    # Checking for missing values
    missing_x = X_train.isnull().sum()
    missing_x_percent = X_train.isnull().mean() * 100

    # Checking for duplicates
    duplicate_designation = X_train.duplicated(subset=["designation"]).sum()
    duplicate_description = X_train[X_train["description"].notna()].duplicated(subset=[
        "description"]).sum()
    duplicate_productid = X_train.duplicated(subset=["productid"]).sum()
    duplicate_imageid = X_train.duplicated(subset=["imageid"]).sum()
    duplicate_prdtypecode = X_train.duplicated(subset=["prdtypecode"]).sum()

    duplicate_designation_percent = X_train.duplicated(
        subset=["designation"]).mean() * 100
    duplicate_description_percent = X_train[X_train["description"].notna(
    )].duplicated(subset=["description"]).mean() * 100
    duplicate_productid_percent = X_train.duplicated(
        subset=["productid"]).mean() * 100
    duplicate_imageid_percent = X_train.duplicated(subset=["imageid"]).mean() * 100
    duplicate_prdtypecode_percent = X_train.duplicated(
        subset=["prdtypecode"]).mean() * 100

    # Unicity
    unique_designation = X_train['designation'].nunique()
    unique_description = X_train['description'].nunique()
    unique_productids = X_train['productid'].nunique()
    unique_imageids = X_train['imageid'].nunique()
    unique_prdtypecode = X_train['prdtypecode'].nunique()

    # Creation of a MultiIndex for the "Check column"
    # Assuming X_train has 'designation', 'description', 'productid', 'imageid', 'prdtypecode'
    index_tuples = [("Missing values", col) for col in ['designation', 'description', 'productid', 'imageid', 'prdtypecode']] + \
                   [("Duplicates", "Designation"), ("Duplicates", "Description"), ("Duplicates", "Productid"),
                    ("Duplicates", "Imageid"), ("Duplicates", "Prdtypecode")] + \
                   [("Unicity", "Designation"), ("Unicity", "Description"), ("Unicity", "Productids"),
                    ("Unicity", "Imageids"), ("Unicity", "Prdtypecode")]

    index = pd.MultiIndex.from_tuples(index_tuples, names=["Check", "Column"])


    # Creation of values and percentages
    values = [
        missing_x["designation"],
        missing_x["description"],
        missing_x["productid"],
        missing_x["imageid"],
        missing_x["prdtypecode"],
        duplicate_designation,
        duplicate_description,
        duplicate_productid,
        duplicate_imageid,
        duplicate_prdtypecode,
        unique_designation,
        unique_description,
        unique_productids,
        unique_imageids,
        unique_prdtypecode
    ]

    percent_values = [
        round(missing_x_percent["designation"], 2),
        round(missing_x_percent["description"], 2),
        round(missing_x_percent["productid"], 2),
        round(missing_x_percent["imageid"], 2),
        round(missing_x_percent["prdtypecode"], 2),
        round(duplicate_designation_percent, 2),
        round(duplicate_description_percent, 2),
        round(duplicate_productid_percent, 2),
        round(duplicate_imageid_percent, 2),
        round(duplicate_prdtypecode_percent, 2),
        round(unique_designation / len(X_train) * 100, 2),
        round(unique_description / len(X_train) * 100, 2),
        round(unique_productids / len(X_train) * 100, 2),
        round(unique_imageids / len(X_train) * 100, 2),
        round(unique_prdtypecode / len(X_train) * 100, 2)
    ]

    # Create the MultiIndex DataFrame
    check_df = pd.DataFrame({
        "Values": values,
        "Values (%)": percent_values
    }, index=index) # Removed [:15] as index_tuples list has exactly 15 elements now

    # Display in Streamlit
    st.title("Data Quality Checks")

    st.dataframe(check_df.style.format(
        {"Values (%)": "{:.2f}%"}), use_container_width=True) # Added % sign to format

    # Show total duplicate rows
    st.markdown(
        f"**The exact number of duplicates by line is:** {X_train.duplicated().sum()}")

    # Header
    st.header("First Analysis Interpretation")

    # Italic text
    st.markdown("""
    *=== Designation ===*

    *No null values but 3% of duplicates, which can cause issues further on.*

    *=== Description ===*

    *35% of missing values, suggesting that descriptions are optional for sellers on Rakuten.*
    *14% of duplicates, which suggests that some sellers may have...*
    *- copy-pasted descriptions for identical products they sold numerous copies of*
    *- copy-pasted descriptions for identical products with some slight feature differences (different color, size, state, etc.)*

    *Missing values and duplicates will require some preprocessing.*

    *=== Productid and Imageid ===*

    *Unique identifiers generated for each product. No missing values or duplicates.*

    *=== Product Type Code ===*

    *There are 27 unique product types. We will drill-down into these further on.*
    """)


# --- Page 2: DataVizualization ---
if page == pages[1]:
    st.header("Product type identification")
    st.subheader("Generating Wordclouds for each product type")

    # Stopwords and replacements
    french_stopwords = set(stopwords.words('french'))
    # --- IMPORTANT: FILL YOUR CUSTOM STOPWORDS HERE ---
    custom_stopwords = {
        'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'à', 'en', 'avec', 'pour', 'sur', 'dans',
        'par', 'plus', 'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles', 'ne', 'pas', 'est', 'sont',
        'au', 'aux', 'qui', 'que', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi', 'code', 'produit',
        'produits', 'photo', 'article', 'articles', 'vendu', 'vente', 'fr', 'comme', 'type', 'tres',
        'nouveau', 'neuf', 'occasion', 'etat', 'voir', 'faire', 'tout', 'tous', 'tres', 'egalement',
        'disponible', 'disponibles', 'grand', 'petit', 'plusieurs', 'differents', 'modele', 'modeles',
        'noir', 'blanc', 'rouge', 'bleu', 'vert', 'jaune', 'rose', 'marron', 'gris', 'argent', 'or',
        'couleur', 'taille', 'longueur', 'largeur', 'hauteur', 'poids', 'matiere', 'plastique', 'metal',
        'bois', 'cuir', 'tissu', 'qualite', 'design', 'utilisation', 'ensemble', 'ideale', 'ideal',
        'parfait', 'parfaite', 'belle', 'beau', 'super', 'original', 'univers', 'collection', 'edition',
        'lot', 'pcs', 'piece', 'pieces', 'paire', 'paires', 'cm', 'mm', 'm', 'g', 'kg', 'ml', 'cl', 'l',
        'marque', 'marqueur', 'marqueurs', 'genre', 'type', 'sous', 'vers', 'avant', 'apres', 'depuis',
        'pendant', 'entre', 'sous', 'devant', 'derriere', 'autour', 'malgre', 'sauf', 'selon', 'sans',
        'sous', 'vers', 'voici', 'voilà', 'depuis', 'chez', 'avec', 'contre', 'entre', 'jusque', 'malgré',
        'moyennant', 'parmi', 'pendant', 'sauf', 'selon', 'sous', 'sur', 'vers', 'via', 'voire'
    }
    all_stopwords = french_stopwords.union(custom_stopwords)

    word_grouping = {
        'livres': 'livre', 'jeux': 'jeu', 'toy': 'jeu', 'jouets': 'jouet', 'enfants': 'enfant',
        'car': 'voiture', 'tools': 'outils', 'chaussette': 'chaussettes', 'bebe': 'bébé', 'telecommande': 'télécommandé',
        'cuisine': 'cuisson', 'four': 'fours', 'maison': 'maisons', 'exterieur': 'extérieur',
        'bain': 'salle de bain', 'lit': 'literie', 'sommier': 'literie', 'matelas': 'literie',
        'chambre': 'literie', 'jardin': 'extérieur', 'piscine': 'extérieur', 'sport': 'sportif',
        'electronique': 'électronique', 'informatique': 'informatique', 'telephonie': 'téléphonie',
        'accessoire': 'accessoires', 'hifi': 'hifi', 'tv': 'télévision', 'son': 'audio', 'image': 'vidéo',
        'dvd': 'film', 'blu-ray': 'film', 'musique': 'musique', 'vinyle': 'vinyle', 'cd': 'musique',
        'art': 'art', 'collection': 'collection', 'timbre': 'collection', 'monnaie': 'collection',
        'billet': 'collection', 'vintage': 'vintage', 'retro': 'rétro', 'ancien': 'ancien', 'antiquite': 'antiquité',
        'mode': 'mode', 'vetement': 'vêtement', 'chaussure': 'chaussures', 'bijoux': 'bijou', 'montre': 'montre',
        'sacs': 'sac', 'bagage': 'bagages', 'beaute': 'beauté', 'sante': 'santé', 'maquillage': 'maquillage',
        'parfum': 'parfum', 'soin': 'soins', 'bien': 'bien-être', 'etre': 'bien-être', 'animal': 'animaux',
        'chien': 'animaux', 'chat': 'animaux', 'oiseau': 'animaux', 'poisson': 'animaux', 'rongeur': 'animaux',
        'aquarium': 'aquarium', 'aquariophilie': 'aquariophilie', 'aliment': 'alimentation', 'boisson': 'boisson',
        'epicerie': 'épicerie', 'vins': 'vin', 'spiritueux': 'spiritueux', 'the': 'thé', 'cafe': 'café',
        'chocolat': 'chocolat', 'bonbon': 'confiserie', 'sucre': 'sucre', 'sel': 'sel', 'poivre': 'poivre',
        'epice': 'épices', 'herbe': 'herbes', 'cuisine': 'cuisine', 'recette': 'recette', 'livre': 'livre',
        'magazine': 'magazine', 'bd': 'bd', 'manga': 'manga', 'comics': 'comics', 'journal': 'journal',
        'revue': 'revue', 'journal': 'journal', 'papeterie': 'papeterie', 'bureau': 'bureau', 'ecriture': 'écriture',
        'dessin': 'dessin', 'peinture': 'peinture', 'sculpture': 'sculpture', 'manuel': 'manuel', 'dictionnaire': 'dictionnaire',
        'encyclopedie': 'encyclopédie', 'scolaire': 'scolaire', 'universitaire': 'universitaire', 'education': 'éducation',
        'formation': 'formation', 'professionnel': 'professionnel', 'musique': 'musique', 'film': 'film',
        'loisir': 'loisirs', 'sport': 'sport', 'salle': 'salle de bain', 'jardin': 'jardin', 'exterieur': 'extérieur',
        'interieur': 'intérieur', 'maison': 'maison', 'deco': 'décoration', 'meuble': 'mobilier',
        'luminaire': 'luminaire', 'linge': 'linge de maison', 'cuisine': 'cuisine', 'salledebain': 'salle de bain',
        'jardinage': 'jardinage', 'outils': 'outils', 'bricolage': 'bricolage', 'voiture': 'voiture',
        'moto': 'moto', 'pieces': 'pièces auto/moto', 'equipement': 'équipement', 'accessoires': 'accessoires auto/moto',
        'pneu': 'pneus', 'jante': 'jantes', 'moteur': 'moteur', 'frein': 'freins', 'suspension': 'suspension',
        'carrosserie': 'carrosserie', 'habitacle': 'habitacle', 'nettoyage': 'nettoyage auto', 'entretien': 'entretien auto',
        'garage': 'garage', 'atelier': 'atelier', 'securite': 'sécurité', 'alarme': 'alarme', 'gps': 'gps',
        'audio': 'audio auto', 'video': 'vidéo auto', 'camping': 'camping', 'randonnee': 'randonnée',
        'voyage': 'voyage', 'bagagerie': 'bagagerie', 'tente': 'tente', 'sacre': 'sac à dos', 'couchage': 'couchage',
        'cuisine': 'cuisine camping', 'chaussures': 'chaussures randonnée', 'vetements': 'vêtements sport',
        'accessoires': 'accessoires sport', 'fitness': 'fitness', 'musculation': 'musculation', 'yoga': 'yoga',
        'course': 'course à pied', 'cyclisme': 'cyclisme', 'natation': 'natation', 'sports': 'sports collectifs',
        'hiver': 'sports d\'hiver', 'glisse': 'sports de glisse', 'nautique': 'sports nautiques', 'raquette': 'sports de raquette',
        'precision': 'sports de précision', 'combat': 'sports de combat', 'arts': 'arts martiaux', 'equitation': 'équitation',
        'chasse': 'chasse', 'peche': 'pêche', 'armes': 'armes', 'munitions': 'munitions', 'optique': 'optique',
        'couteau': 'couteaux', 'lampe': 'lampes', 'survie': 'survie', 'securite': 'sécurité', 'defense': 'défense',
        'auto': 'automobile', 'moto': 'moto', 'bateau': 'bateau', 'avion': 'avion', 'train': 'train',
        'jouets': 'jouet', 'jeux': 'jeu', 'puzzles': 'puzzle', 'construction': 'construction', 'maquette': 'maquette',
        'modelisme': 'modélisme', 'robotique': 'robotique', 'educatif': 'éducatif', 'science': 'science',
        'decouverte': 'découverte', 'experience': 'expérience', 'loisirs': 'loisirs créatifs', 'creatif': 'loisirs créatifs',
        'peinture': 'peinture', 'dessin': 'dessin', 'musique': 'musique', 'instrument': 'instrument de musique',
        'chant': 'chant', 'guitare': 'guitare', 'piano': 'piano', 'batterie': 'batterie', 'clavier': 'clavier',
        'violon': 'violon', 'violoncelle': 'violoncelle', 'flute': 'flute', 'clarinette': 'clarinette', 'saxophone': 'saxophone',
        'trompette': 'trompette', 'harmonica': 'harmonica', 'accordéon': 'accordéon', 'percussion': 'percussions',
        'dj': 'dj', 'sono': 'sono', 'lumiere': 'lumière', 'effet': 'effets spéciaux', 'studio': 'studio',
        'enregistrement': 'enregistrement', 'micro': 'microphone', 'casque': 'casque audio', 'enceinte': 'enceinte',
        'ampli': 'ampli', 'table': 'table de mixage', 'logiciel': 'logiciel musical', 'plug-in': 'plug-in',
        'sample': 'samples', 'boucle': 'boucles', 'sonorisation': 'sonorisation', 'lumiere': 'lumière spectacle',
        'scene': 'scène', 'structure': 'structure', 'accessoires': 'accessoires spectacle', 'decoration': 'décoration événement'
    }

    all_stopwords = french_stopwords.union(custom_stopwords)

    word_grouping = {
        'livres': 'livre', 'jeux': 'jeu', 'toy': 'jeu', 'jouets': 'jouet', 'enfants': 'enfant',
        'car': 'voiture', 'tools': 'outils', 'chaussette': 'chaussettes', 'bebe': 'bébé', 'telecommande': 'télécommandé',
        # Added based on common French terms and product types
        'cuisine': 'cuisson', 'four': 'fours', 'maison': 'maisons', 'exterieur': 'extérieur',
        'bain': 'salle de bain', 'lit': 'literie', 'sommier': 'literie', 'matelas': 'literie',
        'chambre': 'literie', 'jardin': 'extérieur', 'piscine': 'extérieur', 'sport': 'sportif',
        'electronique': 'électronique', 'informatique': 'informatique', 'telephonie': 'téléphonie',
        'accessoire': 'accessoires', 'hifi': 'hifi', 'tv': 'télévision', 'son': 'audio', 'image': 'vidéo',
        'dvd': 'film', 'blu-ray': 'film', 'musique': 'musique', 'vinyle': 'vinyle', 'cd': 'musique',
        'art': 'art', 'collection': 'collection', 'timbre': 'collection', 'monnaie': 'collection',
        'billet': 'collection', 'vintage': 'vintage', 'retro': 'rétro', 'ancien': 'ancien', 'antiquite': 'antiquité',
        'mode': 'mode', 'vetement': 'vêtement', 'chaussure': 'chaussures', 'bijoux': 'bijou', 'montre': 'montre',
        'sacs': 'sac', 'bagage': 'bagages', 'beaute': 'beauté', 'sante': 'santé', 'maquillage': 'maquillage',
        'parfum': 'parfum', 'soin': 'soins', 'bien': 'bien-être', 'etre': 'bien-être', 'animal': 'animaux',
        'chien': 'animaux', 'chat': 'animaux', 'oiseau': 'animaux', 'poisson': 'animaux', 'rongeur': 'animaux',
        'aquarium': 'aquarium', 'aquariophilie': 'aquariophilie', 'aliment': 'alimentation', 'boisson': 'boisson',
        'epicerie': 'épicerie', 'vins': 'vin', 'spiritueux': 'spiritueux', 'the': 'thé', 'cafe': 'café',
        'chocolat': 'chocolat', 'bonbon': 'confiserie', 'sucre': 'sucre', 'sel': 'sel', 'poivre': 'poivre',
        'epice': 'épices', 'herbe': 'herbes', 'cuisine': 'cuisine', 'recette': 'recette', 'livre': 'livre',
        'magazine': 'magazine', 'bd': 'bd', 'manga': 'manga', 'comics': 'comics', 'journal': 'journal',
        'revue': 'revue', 'journal': 'journal', 'papeterie': 'papeterie', 'bureau': 'bureau', 'ecriture': 'écriture',
        'dessin': 'dessin', 'peinture': 'peinture', 'sculpture': 'sculpture', 'manuel': 'manuel', 'dictionnaire': 'dictionnaire',
        'encyclopedie': 'encyclopédie', 'scolaire': 'scolaire', 'universitaire': 'universitaire', 'education': 'éducation',
        'formation': 'formation', 'professionnel': 'professionnel', 'musique': 'musique', 'film': 'film',
        'loisir': 'loisirs', 'sport': 'sport', 'salle': 'salle de bain', 'jardin': 'jardin', 'exterieur': 'extérieur',
        'interieur': 'intérieur', 'maison': 'maison', 'deco': 'décoration', 'meuble': 'mobilier',
        'luminaire': 'luminaire', 'linge': 'linge de maison', 'cuisine': 'cuisine', 'salledebain': 'salle de bain',
        'jardinage': 'jardinage', 'outils': 'outils', 'bricolage': 'bricolage', 'voiture': 'voiture',
        'moto': 'moto', 'pieces': 'pièces auto/moto', 'equipement': 'équipement', 'accessoires': 'accessoires auto/moto',
        'pneu': 'pneus', 'jante': 'jantes', 'moteur': 'moteur', 'frein': 'freins', 'suspension': 'suspension',
        'carrosserie': 'carrosserie', 'habitacle': 'habitacle', 'nettoyage': 'nettoyage auto', 'entretien': 'entretien auto',
        'garage': 'garage', 'atelier': 'atelier', 'securite': 'sécurité', 'alarme': 'alarme', 'gps': 'gps',
        'audio': 'audio auto', 'video': 'vidéo auto', 'camping': 'camping', 'randonnee': 'randonnée',
        'voyage': 'voyage', 'bagagerie': 'bagagerie', 'tente': 'tente', 'sacre': 'sac à dos', 'couchage': 'couchage',
        'cuisine': 'cuisine camping', 'chaussures': 'chaussures randonnée', 'vetements': 'vêtements sport',
        'accessoires': 'accessoires sport', 'fitness': 'fitness', 'musculation': 'musculation', 'yoga': 'yoga',
        'course': 'course à pied', 'cyclisme': 'cyclisme', 'natation': 'natation', 'sports': 'sports collectifs',
        'hiver': 'sports d\'hiver', 'glisse': 'sports de glisse', 'nautique': 'sports nautiques', 'raquette': 'sports de raquette',
        'precision': 'sports de précision', 'combat': 'sports de combat', 'arts': 'arts martiaux', 'equitation': 'équitation',
        'chasse': 'chasse', 'peche': 'pêche', 'armes': 'armes', 'munitions': 'munitions', 'optique': 'optique',
        'couteau': 'couteaux', 'lampe': 'lampes', 'survie': 'survie', 'securite': 'sécurité', 'defense': 'défense',
        'auto': 'automobile', 'moto': 'moto', 'bateau': 'bateau', 'avion': 'avion', 'train': 'train',
        'jouets': 'jouet', 'jeux': 'jeu', 'puzzles': 'puzzle', 'construction': 'construction', 'maquette': 'maquette',
        'modelisme': 'modélisme', 'robotique': 'robotique', 'educatif': 'éducatif', 'science': 'science',
        'decouverte': 'découverte', 'experience': 'expérience', 'loisirs': 'loisirs créatifs', 'creatif': 'loisirs créatifs',
        'peinture': 'peinture', 'dessin': 'dessin', 'musique': 'musique', 'instrument': 'instrument de musique',
        'chant': 'chant', 'guitare': 'guitare', 'piano': 'piano', 'batterie': 'batterie', 'clavier': 'clavier',
        'violon': 'violon', 'violoncelle': 'violoncelle', 'flute': 'flute', 'clarinette': 'clarinette', 'saxophone': 'saxophone',
        'trompette': 'trompette', 'harmonica': 'harmonica', 'accordéon': 'accordéon', 'percussion': 'percussions',
        'dj': 'dj', 'sono': 'sono', 'lumiere': 'lumière', 'effet': 'effets spéciaux', 'studio': 'studio',
        'enregistrement': 'enregistrement', 'micro': 'microphone', 'casque': 'casque audio', 'enceinte': 'enceinte',
        'ampli': 'ampli', 'table': 'table de mixage', 'logiciel': 'logiciel musical', 'plug-in': 'plug-in',
        'sample': 'samples', 'boucle': 'boucles', 'sonorisation': 'sonorisation', 'lumiere': 'lumière spectacle',
        'scene': 'scène', 'structure': 'structure', 'accessoires': 'accessoires spectacle', 'decoration': 'décoration événement'
    }

    # Tokenization and cleaning
    def clean_text(text):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        return [word_grouping.get(w, w) for w in tokens if w not in all_stopwords]

    data = X_train.copy() # Use the already loaded and merged X_train

    # UI
    st.title("Word Clouds and Frequency Tables by Product Type")
    selected_type = st.selectbox(
        "Choose a product type:", sorted(data['prdtypecode'].unique()))

    # Filter and tokenize
    subset = data[data['prdtypecode'] == selected_type]
    designation_tokens = subset['designation'].dropna().astype(
        str).apply(clean_text).sum()
    description_tokens = subset['description'].dropna().astype(
        str).apply(clean_text).sum()
    combined_tokens = designation_tokens + description_tokens

    # Word clouds
    def plot_wordcloud(tokens, title, colormap):
        text = ' '.join(tokens)
        if not text:
            st.warning(f"No valid text for: {title}")
            return
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap=colormap).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16)
        st.pyplot(fig) # Use st.pyplot to display matplotlib figures


    # Layout for word clouds
    st.subheader("Word Clouds")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_wordcloud(designation_tokens, "Designation", "Reds")
    with col2:
        plot_wordcloud(description_tokens, "Description", "Blues")
    with col3:
        plot_wordcloud(combined_tokens, "Combined", "Greens")

    # Frequency tables
    def get_freq_df(tokens):
        return pd.DataFrame(Counter(tokens).most_common(20), columns=["Word", "Frequency"])

    st.subheader("Frequency Tables")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.dataframe(get_freq_df(designation_tokens), use_container_width=True)
    with col5:
        st.dataframe(get_freq_df(description_tokens), use_container_width=True)
    with col6:
        st.dataframe(get_freq_df(combined_tokens), use_container_width=True)


    st.subheader("Defining the product types")
    st.markdown("""Based on the observation of the word frequencies and wordclouds, we are able to define the product types associated with each code.
        The choice of product types is based on a cross-analysis of the most frequent words, qualitative observation of the word clouds, and coherence checks with sample products from each product type number.
        The naming may be challenged and refined in the future.""")

    # Mapping dictionary
    prdtypes = {
        10: "Livres d'occasion",
        40: "Jeux vidéo",
        50: "Accessoires de jeux vidéo",
        60: "Consoles de jeux vidéo",
        1140: "Figurines Enfant",
        1160: "Cartes à Collectionner",
        1180: "Figurines Adulte et Jeux de role",
        1280: "Jouets",
        1281: "Jeux de société",
        1300: "Jouets télécommandés",
        1301: "Chaussettes bébé",
        1302: "Pêche Enfant",
        1320: "Puériculture",
        1560: "Mobilier intérieur",
        1920: "Literie",
        1940: "Alimentation",
        2060: "Décoration",
        2220: "Animaux",
        2280: "Revues et Magazines",
        2403: "Lots Magazines, Livres et BDs",
        2462: "Jeux d'occasion",
        2522: "Papeterie",
        2582: "Mobilier de jardin",
        2583: "Equipement de piscine",
        2585: "Entretien",
        2705: "Livres neufs",
        2905: "Jeux PC"
    }

    # Add the mapping as a new column (X_train is already merged)
    # Y_train["prdtype"] = Y_train["prdtypecode"].map(prdtypes) # Y_train is not used later in this block for display, only X_train
    X_train["prdtype"] = X_train["prdtypecode"].map(prdtypes)


    # Show a sample table
    st.subheader("Sample of Mapped Product Types in X_train")
    st.dataframe(X_train.sample(10).sort_values(
        by="prdtypecode"), use_container_width=True)


    st.subheader("Distribution of products across product types")
    st.write("General statistics")
    prdtypecode_count = X_train['prdtypecode'].value_counts()

    # Key statistics
    prdtypecode_max_frequency = prdtypecode_count.idxmax()
    prdtypecode_min_frequency = prdtypecode_count.idxmin()
    prdtypecode_avg_frequency = round(prdtypecode_count.mean(), 0)
    prdtypecode_med_frequency = round(prdtypecode_count.median(), 0)
    prdtypecode_std_classes = round(prdtypecode_count.std(), 0)
    imbalance_ratio = round(prdtypecode_count.max() / prdtypecode_count.min(), 1)

    # Class names from prdtypes dictionary
    prdtype_max_frequency = prdtypes.get(prdtypecode_max_frequency, "Unknown")
    prdtype_min_frequency = prdtypes.get(prdtypecode_min_frequency, "Unknown")

    # Create DataFrame
    stat_analysis = pd.DataFrame({
        "Statistic": [
            "Total number of observations",
            "# Unique classes",
            "Most frequent class",
            "Max frequency",
            "Least frequent class",
            "Min frequency",
            "Imbalance ratio",
            "Median frequency",
            "Average frequency",
            "Standard deviation of frequencies"
        ],
        "Value": [
            len(X_train),
            X_train['prdtypecode'].nunique(),
            prdtype_max_frequency,
            prdtypecode_count.max(),
            prdtype_min_frequency,
            prdtypecode_count.min(),
            imbalance_ratio,
            prdtypecode_med_frequency,
            prdtypecode_avg_frequency,
            prdtypecode_std_classes
        ]
    })

    # Streamlit UI
    st.subheader("General statistics")
    st.dataframe(stat_analysis, use_container_width=True)


    st.write("Distribution")

    # Your original code starts here
    prdtypecode_count_index = prdtypecode_count.index
    prdtypecode_proportions = prdtypecode_count / len(X_train) * 100
    prdtype_sorted_list = [prdtypes[code] for code in prdtypecode_count_index]

    # Create the first plot
    st.header("Proportion and Occurrences of Each Product Type")
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    sns.barplot(
        x=prdtype_sorted_list,
        y=prdtypecode_proportions[prdtypecode_count_index],
        order=prdtype_sorted_list,
        color="lightblue",
        ax=ax1
    )
    ax1.set_ylabel('Proportion (%)')
    ax1.set_xlabel('Product Type')
    ax1.set_title('Proportion and Occurences of Each Product Type',
                  fontweight='bold')
    ax1.grid(True, axis='y', color='grey', linewidth=0.5, linestyle=':')
    ax1.tick_params(axis='x', rotation=90)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Occurrences')
    ax2.set_ylim(0, prdtypecode_count[prdtypecode_count_index].max())

    st.pyplot(fig1)

    # Create the table
    st.header("Distribution of products per product Type Code")
    prdtypecode_count_df = pd.DataFrame({
        "Product Type": prdtype_sorted_list,
        "Occurences": prdtypecode_count
    })
    st.dataframe(prdtypecode_count_df)

    # Create the boxplot
    st.header("Dispersion Product Types per # Occurences")
    fig2 = plt.figure(figsize=(14, 3))
    sns.boxplot(x=prdtypecode_count.values, color="lightblue")
    plt.title("Dispersion Product Types per # Occurences",
              fontsize=14, fontweight='bold')
    plt.xlabel("Occurrences", fontsize=12)
    st.pyplot(fig2)

    st.markdown("""
    > We can see that **50% of products** have between **1,500** and **5,000 occurrences**.
    > **Équipement de piscine** is an **outlier** — it appears more than **10,200 times**.
    > This may cause **overfitting**, so we will need to account for its **over-representation** in future steps.
    """)


    st.write("Data inspection")
    # Calculate the lengths of designation and description
    # X_train already loaded with prdtype, ensure original columns are there
    # These operations will be re-run on every interaction, consider caching if they become slow
    X_train['designation_length'] = X_train['designation'].astype(str).str.len()
    X_train['description_length'] = X_train['description'].astype(str).str.len()

    # Group by 'prdtypecode' and count the null values in 'description' and total values
    null_counts = X_train.groupby('prdtypecode').agg({
        'description': lambda x: x.isnull().sum(),
        'designation': 'count' # This counts non-null designations
    }).reset_index()

    # Map 'prdtypecode' to 'prdtype' using the prdtypes dictionary
    null_counts['prdtype'] = null_counts['prdtypecode'].map(prdtypes)

    # Calculate the percentage of null values
    null_counts['null_descriptions_pct'] = round(
        (null_counts['description'] / null_counts['designation']) * 100, 1)

    # Rename the columns for clarity
    null_counts = null_counts.rename(columns={
        'description': 'null_descriptions_count',
        'designation': 'total_count'
    })

    # Select the desired columns
    final_df = null_counts[['prdtypecode', 'prdtype',
                            'null_descriptions_count', 'null_descriptions_pct']]

    # Sort the DataFrame by the number of null values in 'description' in descending order
    final_df = final_df.sort_values('null_descriptions_count', ascending=False)

    # Streamlit display
    st.title("Number of Null Values in Description per Product Type")

    # Create and display the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=final_df, x='prdtype',
                y='null_descriptions_count', ax=ax1, color='skyblue')
    ax1.set_title('Number of Null Values in Description per Product Type')
    ax1.set_xlabel('Product Type')
    ax1.set_ylabel('Number of Null Values')
    ax1.set_xticks(range(len(final_df['prdtype'])))
    ax1.set_xticklabels(final_df['prdtype'], rotation=90, ha='right')
    plt.grid(True, linestyle=':', alpha=0.7)

    ax2 = ax1.twinx()
    sns.lineplot(data=final_df, x='prdtype', y='null_descriptions_pct',
                 ax=ax2, color='lightcoral', marker='o')
    ax2.set_ylabel('Percentage of Null Values (%)')

    plt.tight_layout()
    st.pyplot(fig) # Use st.pyplot

    # Display the final DataFrame sorted by percentage of null values
    final_df_sorted = final_df.sort_values(
        'null_descriptions_pct', ascending=False)
    st.header("Number of Null Values in Description per Product Type")
    st.dataframe(final_df_sorted)

    # Display the analysis text
    st.write('''We notice that 3 product types have >40% of null descriptions. 3 of these categories relate to books or magazines,
             suggesting that the title was a sufficient source of information for sellers and buyers, especially when it is second hand.
             We will have to take this in consideration in the future to prevent underfitting.''')


    st.write("Replicate descriptions")
    # Create a new column that checks if designation is identical to description
    X_train['identical_designation_description'] = X_train['designation'].astype(str) == X_train['description'].astype(str)

    # Group by 'prdtypecode' and count the occurrences where designation is identical to description
    identical_counts = X_train.groupby('prdtypecode').agg({
        'identical_designation_description': 'sum',
        'designation': 'count'
    }).reset_index()

    # Map 'prdtypecode' to 'prdtype' using the prdtypes dictionary
    identical_counts['prdtype'] = identical_counts['prdtypecode'].map(prdtypes)

    # Calculate the percentage of identical designation and description
    identical_counts['identical_pct'] = round(
        (identical_counts['identical_designation_description'] / identical_counts['designation']) * 100, 1)

    # Rename the columns for clarity
    identical_counts = identical_counts.rename(columns={
        'identical_designation_description': 'identical_count',
        'designation': 'total_count'
    })

    # Select the desired columns
    final_df = identical_counts[['prdtypecode',
                                 'prdtype', 'identical_count', 'identical_pct']]

    # Sort the DataFrame by the number of identical values in descending order
    final_df = final_df.sort_values('identical_count', ascending=False)

    # Streamlit display
    st.title("Number of Products with Identical Designation and Description per Product Type")

    # Create and display the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=final_df, x='prdtype',
                y='identical_count', ax=ax1, color='skyblue')
    ax1.set_title(
        'Number of Products with Identical Designation and Description per Product Type')
    ax1.set_xlabel('Product Type')
    ax1.set_ylabel('Number of Identical Values')
    ax1.set_xticks(range(len(final_df['prdtype'])))
    ax1.set_xticklabels(final_df['prdtype'], rotation=90, ha='right')
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2 = ax1.twinx()
    sns.lineplot(data=final_df, x='prdtype', y='identical_pct',
                 ax=ax2, color='lightcoral', marker='o')
    ax2.set_ylabel('Percentage of Identical Values (%)')

    plt.tight_layout()
    st.pyplot(fig) # Use st.pyplot

    # Display the final DataFrame sorted by percentage of identical values
    final_df_sorted = final_df.sort_values('identical_pct', ascending=False)
    st.header(
        "Number of Products with Identical Designation and Description per Product Type")
    st.dataframe(final_df_sorted)

    # Display sample of products with identical designation and description
    replicate_products = X_train[X_train['identical_designation_description']]
    replicate_products_count = replicate_products.shape[0]

    st.header("Random Sample of 10 Products with Identical Designation and Description")
    st.write(f'There are {replicate_products_count} replicate products in total.')
    st.dataframe(replicate_products[[
        'prdtypecode', 'prdtype', 'designation', 'description']].sample(10))

    # Display the analysis text
    st.write('''Replicate dimensions is a limited phenomenon. It reaches 1.4% of products for Puériculture products (child care).
             We will still have this in mind for the preprocessing.''')


    st.header("Duplicate designations and descriptions")

    st.title("Duplicate Values Analysis")

    # DATA PROCESSING (NON-STREAMLIT CODE) - This section seems fine as is for calculations
    # Calculate the lengths of designation and description (already done above, but safe to re-run if needed)
    # X_train['designation_length'] = X_train['designation'].str.len()
    # X_train['description_length'] = X_train['description'].str.len()

    # Group by 'prdtypecode' and count duplicates
    duplicate_counts = X_train.groupby('prdtypecode').agg(
        designation_duplicates=pd.NamedAgg(
            column='designation', aggfunc=lambda x: x.dropna().duplicated(keep=False).sum()),
        description_duplicates=pd.NamedAgg(
            column='description', aggfunc=lambda x: x.dropna().duplicated(keep=False).sum()),
        designation_non_null_count=pd.NamedAgg(
            column='designation', aggfunc=lambda x: x.dropna().count()),
        description_non_null_count=pd.NamedAgg(
            column='description', aggfunc=lambda x: x.dropna().count())
    ).reset_index()

    # Map 'prdtypecode' to 'prdtype'
    duplicate_counts['prdtype'] = duplicate_counts['prdtypecode'].map(prdtypes)

    # Calculate percentages
    duplicate_counts['duplicate_designations_pct'] = round(
        (duplicate_counts['designation_duplicates'] / duplicate_counts['designation_non_null_count']) * 100, 1)
    duplicate_counts['duplicate_descriptions_pct'] = round(
        (duplicate_counts['description_duplicates'] / duplicate_counts['description_non_null_count']) * 100, 1)

    # Rename columns
    duplicate_counts = duplicate_counts.rename(columns={
        'designation_duplicates': 'duplicate_designations_count',
        'description_duplicates': 'duplicate_descriptions_count',
    })

    # Create final dataframe
    final_df = duplicate_counts[['prdtypecode', 'prdtype', 'duplicate_designations_count',
                                 'duplicate_designations_pct', 'duplicate_descriptions_count',
                                 'duplicate_descriptions_pct']]

    # Sort and calculate means
    final_df = final_df.sort_values(
        'duplicate_designations_count', ascending=False)
    mean_designation_count = final_df['duplicate_designations_count'].mean()
    mean_designation_pct = final_df['duplicate_designations_pct'].mean()
    mean_description_count = final_df['duplicate_descriptions_count'].mean()
    mean_description_pct = final_df['duplicate_descriptions_pct'].mean()

    # VISUALIZATION 1: DESIGNATION DUPLICATES
    st.header(
        "Number of Duplicate Values in Designation per Product Type (Excluding Nulls)")
    fig1, ax1 = plt.subplots(figsize=(14, 6))

    # Bar plot
    sns.barplot(data=final_df, x='prdtype', y='duplicate_designations_count',
                ax=ax1, color='skyblue')
    ax1.set_xlabel('Product Type')
    ax1.set_ylabel('Number of Duplicate Values')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Mean line
    ax1.axhline(mean_designation_count, color='skyblue',
                linestyle=':', linewidth=1)
    ax1.text(x=len(final_df)-0.5, y=mean_designation_count,
             s=f'Mean: {mean_designation_count:.1f}', color='skyblue',
             ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7))

    # Percentage line
    ax2 = ax1.twinx()
    sns.lineplot(data=final_df, x='prdtype', y='duplicate_designations_pct',
                 ax=ax2, color='lightcoral', marker='o')
    ax2.set_ylabel('Percentage of Duplicate Values (%)')
    ax2.axhline(mean_designation_pct, color='lightcoral',
                linestyle=':', linewidth=1)
    ax2.text(x=len(final_df)-0.5, y=mean_designation_pct,
             s=f'Mean (%): {mean_designation_pct:.1f}%', color='lightcoral',
             ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7))

    st.pyplot(fig1) # Use st.pyplot

    # VISUALIZATION 2: DESCRIPTION DUPLICATES
    st.header(
        "Number of Duplicate Values in Description per Product Type (Excluding Nulls)")
    fig2, ax3 = plt.subplots(figsize=(14, 6))

    # Bar plot
    sns.barplot(data=final_df, x='prdtype', y='duplicate_descriptions_count',
                ax=ax3, color='skyblue')
    ax3.set_xlabel('Product Type')
    ax3.set_ylabel('Number of Duplicate Values')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90, ha='right')
    ax3.grid(True, linestyle=':', alpha=0.7)

    # Mean line
    ax3.axhline(mean_description_count, color='skyblue',
                linestyle=':', linewidth=1)
    ax3.text(x=len(final_df)-0.5, y=mean_description_count,
             s=f'Mean: {mean_description_count:.1f}', color='skyblue',
             ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7))

    # Percentage line
    ax4 = ax3.twinx()
    sns.lineplot(data=final_df, x='prdtype', y='duplicate_descriptions_pct',
                 ax=ax4, color='lightcoral', marker='o')
    ax4.set_ylabel('Percentage of Duplicate Values (%)')
    ax4.axhline(mean_description_pct, color='lightcoral',
                linestyle=':', linewidth=1)
    ax4.text(x=len(final_df)-0.5, y=mean_description_pct,
             s=f'Mean (%): {mean_description_pct:.1f}%', color='lightcoral',
             ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7))

    st.pyplot(fig2) # Use st.pyplot

    # DATA TABLE
    st.header("Duplicate Values Summary")
    final_df_sorted = final_df.sort_values(
        'duplicate_descriptions_pct', ascending=False)
    st.dataframe(final_df_sorted)


    # TITLE AND HEADER
    st.title("Product Duplicates Analysis")
    st.header("Product Types by Percentage of Duplicate Designations vs. Descriptions")

    # CREATE THE VISUALIZATION
    fig = plt.figure(figsize=(10, 12))

    # Scatter plot with your exact parameters
    sns.scatterplot(
        data=final_df,
        x='duplicate_designations_pct',
        y='duplicate_descriptions_pct',
        hue='prdtype',
        size='duplicate_designations_count',
        sizes=(50, 500),
        alpha=0.7
    )

    plt.title('Product Types by Percentage of Duplicate Designations vs. Descriptions')
    plt.xlabel('Percentage of Duplicate Designations (%)')
    plt.ylabel('Percentage of Duplicate Descriptions (%)')
    plt.grid(None, linestyle=':', alpha=0.7)

    # Add annotations
    for i, row in final_df.iterrows():
        plt.text(
            row['duplicate_designations_pct'] + 0.5,
            row['duplicate_descriptions_pct'] + 0.5,
            row['prdtype'],
            fontsize=8,
            alpha=0.8
        )

    # Configure legend
    plt.legend(
        title='Product Type',
        bbox_to_anchor=(0.5, -0.15),
        loc='upper center',
        ncol=3
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Display in Streamlit
    st.pyplot(fig) # Use st.pyplot


    # Display final dataframe
    st.header("Number of Duplicate Values in Designation and Description per Product Type (Excluding Nulls)")
    st.dataframe(final_df)

    # Create two columns for the tables
    col1, col2 = st.columns(2)

    # Get and display top 20 duplicated designations
    with col1:
        st.subheader("Top 20 Most Duplicated Designations")
        top_designations = X_train[X_train.duplicated(
            subset=['designation'], keep=False)]
        top_designations = top_designations.groupby(
            'designation').size().reset_index(name='duplicate_count')
        top_designations = top_designations.merge(
            X_train[['designation', 'prdtypecode']].drop_duplicates(),
            on='designation',
            how='left'
        )
        top_designations['prdtype'] = top_designations['prdtypecode'].map(prdtypes)
        st.dataframe(
            top_designations.sort_values('duplicate_count', ascending=False)
            .head(20)[['prdtypecode', 'prdtype', 'designation', 'duplicate_count']]
        )

    # Get and display top 20 duplicated descriptions
    with col2:
        st.subheader("Top 20 Most Duplicated Descriptions")
        top_descriptions = X_train[X_train.duplicated(
            subset=['description'], keep=False)]
        top_descriptions = top_descriptions.groupby(
            'description').size().reset_index(name='duplicate_count')
        top_descriptions = top_descriptions.merge(
            X_train[['description', 'prdtypecode']].drop_duplicates(),
            on='description',
            how='left'
        )
        top_descriptions['prdtype'] = top_descriptions['prdtypecode'].map(prdtypes)
        st.dataframe(
            top_descriptions.sort_values('duplicate_count', ascending=False)
            .head(20)[['prdtypecode', 'prdtype', 'description', 'duplicate_count']] # Completed the line
        )