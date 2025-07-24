from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from IPython.display import display
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
from scipy.stats import entropy, kstest, linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import mean_squared_error, f1_score, classification_report, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2
from wordcloud import WordCloud
import ast
import base64
import cv2
import imagehash
import io
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import spacy
import streamlit as st
import time
import warnings
import xgboost as xgb


# data import
path = r"./data/raw"
X_test = pd.read_parquet(f"{path}/X_test_update.parquet")
X_train = pd.read_parquet(f"{path}/X_train_update.parquet")
Y_train = pd.read_parquet(f"{path}/Y_train_CVw08PX.parquet")

st.title("Rakuten e-commerce project")
st.sidebar.title("Table of contents")
pages = ["Data Processing", "DataVizualization", "Modelling"]
page = st.sidebar.radio("Go to", pages)


# ----- Page 1: -----
if page == pages[0]:
    st.write("""Display the original train data""")

    st.write("### Presentation of data")

    tables = [('X_train', X_train), ('X_test', X_test), ('Y_train', Y_train)]
    for table_name, table in tables:
        print("\n"+"=" * 120)
        print(f"\n{table_name}\n")
        display(table.head())
        display(table.info())
        print(f"\nDuplicates:\n{table.duplicated().sum()}")

    X_train = X_train.merge(Y_train, how='left', left_index=True,
                            right_index=True, suffixes=('_X_train', '_Y_train'))


    st.write("### The training data")
    st.dataframe(X_train.head())
    # st.write(X_train.shape)
    st.write("### The dataset description")
    st.dataframe(X_train.describe())

    # st.dataframe(Y_train.head())
    # st.write(Y_train.shape)
    # st.dataframe(Y_train.describe())

    if st.checkbox("Show NA"):
        st.dataframe(df.isna().sum())


    # display(X_train.head(10))
    # st.write(X_train.head(10))


    st.write(""""In each column, we are going to investigate: 
    1. Missing values 
    2. Duplicates
    3. Unique modalities""")

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
    index = pd.MultiIndex.from_tuples(
        [("Missing values", col) for col in X_train.columns] +
        [("Duplicates", "Designation"), ("Duplicates", "Description"), ("Duplicates", "Productid"),
        ("Duplicates", "Imageid"), ("Duplicates", "Prdtypecode")] +
        [("Unicity", "Designation"), ("Unicity", "Description"), ("Unicity", "Productids"),
        ("Unicity", "Imageids"), ("Unicity", "Prdtypecode")],
        names=["Check", "Column"]
    )

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
    }, index=index[:15])  # there is an issue here

    # Display in Streamlit
    st.title("Data Quality Checks")

    st.dataframe(check_df.style.format(
        {"Values (%)": "{:.2f}"}), use_container_width=True)

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


# ----- Page 2: -----
# if page == "DataVizualization":
if page == pages[1]:
    st.write("""Product type identification""")
    st.write("""Generating a Wordclouds for each product type""")

    # Download stopwords
    nltk.download('stopwords')

    # Stopwords and replacements
    french_stopwords = set(stopwords.words('french'))
    custom_stopwords = {...}  # ← Your custom stopwords set (as in your notebook)
    all_stopwords = french_stopwords.union(custom_stopwords)

    word_grouping = {
        'livres': 'livre', 'jeux': 'jeu', 'toy': 'jeu', 'jouets': 'jouet', 'enfants': 'enfant',
        'car': 'voiture', 'tools': 'outils'
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

    # Load data


    @st.cache_data
    def load_data():
        return X_train.copy()  # or pd.read_csv("your_file.csv")


    data = load_data()

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
            return st.warning(f"No valid text for: {title}")
        wc = WordCloud(width=800, height=400, background_color="white",
                    colormap=colormap).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16)
        st.pyplot(fig)


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


    st.subheader("""Defining the product types""")
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

    # Add the mapping as a new column
    Y_train["prdtype"] = Y_train["prdtypecode"].map(prdtypes)
    X_train["prdtype"] = X_train["prdtypecode"].map(prdtypes)

    # Show a sample table
    st.subheader("Sample of Mapped Product Types in X_train")
    st.dataframe(X_train.sample(10).sort_values(
        by="prdtypecode"), use_container_width=True)


    st.write("""Distribution of products across product types""")
    st.write("""General statistics""")
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


    st.write("""Distribution""")

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


    st.write("""Data inspection""")
    # Calculate the lengths of designation and description
    X_train['designation_length'] = X_train['designation'].str.len()
    X_train['description_length'] = X_train['description'].str.len()

    # Group by 'prdtypecode' and count the null values in 'description' and total values
    null_counts = X_train.groupby('prdtypecode').agg({
        'description': lambda x: x.isnull().sum(),
        'designation': 'count'
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
    st.pyplot(fig)

    # Display the final DataFrame sorted by percentage of null values
    final_df_sorted = final_df.sort_values(
        'null_descriptions_pct', ascending=False)
    st.header("Number of Null Values in Description per Product Type")
    st.dataframe(final_df_sorted)

    # Display the analysis text
    st.write('''We notice that 3 product types have >40% of null descriptions. 3 of these categories relate to books or magazines, 
            suggesting that the title was a sufficient source of information for sellers and buyers, especially when it is second hand. 
            We will have to take this in consideration in the future to prevent underfitting.''')


    st.write("""Replicate descriptions""")
    # Create a new column that checks if designation is identical to description
    X_train['identical_designation_description'] = X_train['designation'] == X_train['description']

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
    st.pyplot(fig)

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


    st.header("""Duplicate designations and descriptions""")

    st.title("Duplicate Values Analysis")

    # DATA PROCESSING (NON-STREAMLIT CODE)
    # Calculate the lengths of designation and description
    X_train['designation_length'] = X_train['designation'].str.len()
    X_train['description_length'] = X_train['description'].str.len()

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

    st.pyplot(fig1)

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

    st.pyplot(fig2)

    # DATA TABLE
    st.header("Duplicate Values Summary")
    final_df_sorted = final_df.sort_values(
        'duplicate_descriptions_pct', ascending=False)
    st.dataframe(final_df_sorted)


    # PAGE CONFIG MUST BE THE VERY FIRST STREAMLIT COMMAND
    # (nothing can execute before this, not even print statements)
    # TITLE AND HEADER
    st.title("Product Duplicates Analysis")
    st.header("Product Types by Percentage of Duplicate Designations vs. Descriptions")

    # CREATE DUMMY DATA IF final_df DOESN'T EXIST
    # (replace this with your actual data loading code)
    if 'final_df' not in globals():
        data = {
            'prdtype': ['Literie', 'Chaussettes bébé', 'Equipement de piscine',
                        'Jeux vidéo', 'Papeterie', 'Pêche Enfant'],
            'duplicate_designations_pct': [15.0, 35.2, 7.2, 2.1, 2.0, 12.0],
            'duplicate_descriptions_pct': [38.2, 34.8, 34.7, 34.2, 32.2, 16.4],
            'duplicate_designations_count': [647, 284, 737, 53, 99, 299]
        }
        final_df = pd.DataFrame(data)

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
    st.pyplot(fig)


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
            .head(20)[['prdtypecode', 'prdtype', 'description', 'duplicate_count']]
        )


    # Analysis text
    st.subheader("Analysis")
    st.write('''
    Several observations: 
    - Duplicate designations represent c. 5% of products in each product category on average. This is a moderate and predictable phenomenon on an e-commerce platform. 
    - Duplicate descriptions represent c. 13% of products in each product category on average, with few peaks at > 30%. This is a significant phenomenon.
    ''')


    st.header("""Designation and description length analysis""")
    st.header("""Designation and description lengths overview""")

    if 'X_train' not in locals():
        np.random.seed(42)
        X_train = pd.DataFrame({
            'designation': ['Sample text ' * np.random.randint(1, 20) for _ in range(1000)],
            'description': ['Longer sample text ' * np.random.randint(1, 50) for _ in range(1000)]
        })

    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Histograms", "Boxplots"])

    with tab1:
        st.header("Distribution of Text Lengths (Histograms)")

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot the distribution of designation lengths
        sns.histplot(X_train['designation'].str.len(), bins=50,
                    kde=True, color='lightcoral', ax=ax1)
        ax1.set_title('Distribution of Designation Lengths')
        ax1.set_xlabel('Designation Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot the distribution of description lengths
        sns.histplot(X_train['description'].str.len(),
                    bins=50, kde=True, color='skyblue', ax=ax2)
        ax2.set_title('Distribution of Description Lengths')
        ax2.set_xlabel('Description Length (characters)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Show insights
        st.subheader("Insights:")
        st.write('''
        - Designation length has two peaks at 50 words and 100 words. Results are skewed to the left, which is coherent with sellers' natural bias to choose short designations. 
        - The absence of null designations suggests a minimum number of words.
        - Descriptions are in majority <500 words short.
        ''')

    with tab2:
        st.header("Distribution of Text Lengths (Boxplots)")

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the boxplot of designation lengths
        sns.boxplot(y=X_train['designation'].str.len(), color='lightcoral', ax=ax1)
        ax1.set_title('Distribution of Designation Lengths')
        ax1.set_ylabel('Designation Length (# characters)')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot the boxplot of description lengths
        sns.boxplot(y=X_train['description'].str.len(), color='skyblue', ax=ax2)
        ax2.set_title('Distribution of Description Lengths')
        ax2.set_ylabel('Description Length (# characters)')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Show insights
        st.subheader("Insights:")
        st.write('''
        The boxplots provide a different perspective on the same data:
        - The median and quartile values are clearly visible
        - Outliers in the description lengths are more apparent
        - The compact nature of designation lengths is more obvious in this view
        ''')


    st.header("""Designation and description lengths by product type""")
    st.header("""Designation and description length correlation analysis""")

    # Calculate the average lengths
    avg_lengths = X_train.groupby('prdtypecode').agg({
        'designation': lambda x: x.str.len().mean(),
        'description': lambda x: x.str.len().mean()
    }).round(0)

    avg_lengths.columns = ['avg_designation_length', 'avg_description_length']
    avg_lengths = avg_lengths.sort_values(
        'avg_designation_length', ascending=False)
    avg_lengths = avg_lengths.reset_index()
    avg_lengths['prdtype'] = avg_lengths['prdtypecode'].map(prdtypes)

    # Reorder columns
    columns = avg_lengths.columns.tolist()
    columns.insert(1, columns.pop(columns.index('prdtype')))
    avg_lengths = avg_lengths[columns]

    # Calculate summary statistics
    overall_avg_designation = X_train['designation'].str.len().mean()
    overall_avg_description = X_train['description'].str.len().mean()
    unique_types = X_train['prdtypecode'].nunique()

    ### Display the table ###
    st.header("Average Length of Designation and Description per Product Type")
    st.write("=" * 65)  # Divider line
    st.dataframe(avg_lengths.reset_index(drop=True))

    ### Display the graph ###
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot designation length
    color1 = 'lightcoral'
    ax1.set_xlabel('Product Type')
    ax1.set_ylabel('Average Designation Length (characters)', color=color1)
    bars1 = ax1.bar(range(len(avg_lengths)), avg_lengths['avg_designation_length'],
                    color=color1, alpha=0.7, label='Designation Length')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Plot description length
    ax2 = ax1.twinx()
    color2 = 'skyblue'
    ax2.set_ylabel('Average Description Length (characters)', color=color2)
    bars2 = ax2.bar(range(len(avg_lengths)), avg_lengths['avg_description_length'],
                    color=color2, alpha=0.7, width=0.6, label='Description Length')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Set x-axis labels
    ax1.set_xticks(range(len(avg_lengths)))
    ax1.set_xticklabels(avg_lengths['prdtype'], rotation=90, ha='right')
    ax1.set_title('Average Designation and Description Length by Product Type')

    # Add average lines and labels
    ax1.axhline(y=overall_avg_designation, color=color1,
                linestyle='--', label='Overall Avg Designation')
    ax2.axhline(y=overall_avg_description, color=color2,
                linestyle='--', label='Overall Avg Description')

    ax1.text(len(avg_lengths) - 0.5, overall_avg_designation,
            f' Avg: {overall_avg_designation:.0f}', color=color1,
            va='center', ha='right', backgroundcolor='white')
    ax2.text(len(avg_lengths) - 0.5, overall_avg_description,
            f' Avg: {overall_avg_description:.0f}', color=color2,
            va='center', ha='right', backgroundcolor='white')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(None)
    ax2.grid(None)

    plt.tight_layout()
    st.pyplot(fig)

    ### Display summary statistics and analysis ###
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg designation length", f"{overall_avg_designation:.2f} chars")
    with col2:
        st.metric("Avg description length", f"{overall_avg_description:.2f} chars")
    with col3:
        st.metric("Unique product types", unique_types)


    st.write('''
    The distribution of designation length across product types seems fairly homogenous. 
    However, there is much more heterogeneity for description length, with strong outliers. 
    For instance, PC games have very long descriptions. This would require more investigation.
    ''')


    # Set up the Streamlit page
    st.header("Product Text Length Distributions by Category")

    # Get unique product codes
    unique_prdtypecodes = X_train['prdtypecode'].unique()

    # Create a dropdown selector for product categories
    selected_code = st.selectbox(
        "Select a Product Category:",
        options=unique_prdtypecodes,
        format_func=lambda x: f"{prdtypes.get(x, 'Unknown')} (Code: {x})"
    )

    # Filter data for selected category
    subset = X_train[X_train['prdtypecode'] == selected_code]
    prdtype = prdtypes.get(selected_code, "Unknown")

    # Create the three visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Designation Length Distribution
    sns.histplot(subset['designation'].dropna().str.len(),
                bins=50, kde=True, color='lightcoral', ax=ax1)
    ax1.set_title(
        f'Designation Length Distribution\n{prdtype} (Code: {selected_code})')
    ax1.set_xlabel('Designation Length (characters)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Description Length Distribution
    sns.histplot(subset['description'].dropna().str.len(),
                bins=50, kde=True, color='skyblue', ax=ax2)
    ax2.set_title(
        f'Description Length Distribution\n{prdtype} (Code: {selected_code})')
    ax2.set_xlabel('Description Length (characters)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Combined Boxplot
    sns.boxplot(data=[subset['designation'].dropna().str.len(), subset['description'].dropna().str.len()],
                palette=['lightcoral', 'skyblue'], ax=ax3)
    ax3.set_title(f'Length Comparison\n{prdtype} (Code: {selected_code})')
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Designation', 'Description'])
    ax3.set_ylabel('Length (characters)')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and display
    plt.tight_layout()
    st.pyplot(fig)

    # Show summary statistics
    st.subheader(f"Summary Statistics for {prdtype}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Designation Length**")
        st.write(subset['designation'].str.len().describe().to_frame().T)

    with col2:
        st.markdown("**Description Length**")
        st.write(subset['description'].str.len().describe().to_frame().T)


    st.header("""Designation and description length correlation analysis""")
    # Set up the Streamlit page
    st.title("Product Text Length Relationship Analysis")
    st.markdown(
        "### Scatter plots showing designation vs description length for each product category")

    # Calculate text lengths
    X_train['designation_length'] = X_train['designation'].str.len()
    X_train['description_length'] = X_train['description'].str.len()

    # Get unique product codes
    unique_prdtypecodes = sorted(X_train['prdtypecode'].unique())

    # Add controls
    st.sidebar.header("Plot Controls")
    cols_per_row = st.sidebar.slider("Columns per row", 2, 6, 3)
    plot_height = st.sidebar.slider("Plot height (inches)", 3, 8, 5)
    show_stats = st.sidebar.checkbox("Show regression statistics", True)

    # Calculate grid dimensions
    n = len(unique_prdtypecodes)
    n_cols = cols_per_row
    n_rows = int(np.ceil(n / n_cols))

    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5*n_cols, plot_height*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])  # Ensure axes is always 2D array
    axes = axes.flatten()

    # Generate plots
    for i, prdtypecode in enumerate(unique_prdtypecodes):
        ax = axes[i]
        subset = X_train[X_train['prdtypecode'] == prdtypecode]
        prdtype = prdtypes.get(prdtypecode, "Unknown")

        # Clean data
        subset = subset.dropna(subset=['designation_length', 'description_length'])

        if len(subset) > 1:
            # Calculate regression
            x = subset['designation_length']
            y = subset['description_length']
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            r_squared = r_value**2

            # Create scatter plot with regression line
            sns.regplot(x=x, y=y,
                        scatter_kws={'alpha': 0.5, 's': 30, 'color': 'steelblue'},
                        line_kws={'color': 'red', 'linewidth': 2},
                        ci=None, ax=ax)

            # Format plot
            ax.set_title(f"{prdtype} ({prdtypecode})", fontsize=10)
            ax.set_xlabel("Designation Length", fontsize=8)
            ax.set_ylabel("Description Length", fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.3)

            # Add statistics if enabled
            if show_stats:
                stats_text = f"R² = {r_squared:.2f}\ny = {slope:.2f}x + {intercept:.2f}"
                ax.text(0.05, 0.95, stats_text,
                        transform=ax.transAxes,
                        ha='left', va='top',
                        bbox=dict(facecolor='white', alpha=0.8),
                        fontsize=8)
        else:
            ax.set_title(f"{prdtype} ({prdtypecode})", fontsize=10)
            ax.text(0.5, 0.5, "Not enough data",
                    ha='center', va='center',
                    transform=ax.transAxes)

    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    # Show summary
    st.success(f"Displayed {len(unique_prdtypecodes)} product categories")


    st.header("""Image inspection""")
    # Set up the Streamlit app
    st.title("Image Metadata Analyzer")
    st.markdown("### Analyze image properties from directory")


    # Path configuration
    IMAGE_DATA_PATH = st.text_input("Enter image directory path:",
                                    './data/raw/images_bgrm/')
    image_directory = os.path.join(IMAGE_DATA_PATH, 'image_train')

    if not os.path.exists(image_directory):
        st.error("Directory not found! Please check the path.")
        st.stop()

    # Image processing function


    def process_image(image_file_name):
        pattern = re.compile(r'image_(\d+)_product_(\d+)\.jpg')
        match = pattern.match(image_file_name)
        if match:
            imageid = match.group(1)
            productid = match.group(2)
            full_image_path = os.path.join(image_directory, image_file_name)

            try:
                with Image.open(full_image_path) as img:
                    width, height = img.size
                    img_format = img.format
                    size_bytes = os.path.getsize(full_image_path)
                    size_kb = round(size_bytes / 1024, 2)

                    return {
                        'productid': productid,
                        'imageid': imageid,
                        'imagepath': full_image_path,
                        'width': width,
                        'height': height,
                        'format': img_format,
                        'size_bytes': size_bytes,
                        'size_kb': size_kb
                    }
            except Exception as e:
                st.warning(f"Could not process {image_file_name}: {str(e)}")
                return None


    # Process images with progress bar
    if st.button("Analyze Images"):
        with st.spinner("Processing images..."):
            image_files = [f for f in os.listdir(
                image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            progress_bar = st.progress(0)

            # Process images (without ThreadPoolExecutor for simplicity)
            results = []
            for i, image_file in enumerate(image_files):
                result = process_image(image_file)
                if result:
                    results.append(result)
                progress_bar.progress((i + 1) / len(image_files))

            df = pd.DataFrame(results)

            if not df.empty:
                # Calculate unique sizes and formats
                unique_sizes = set(zip(df['width'], df['height']))
                unique_formats = set(df['format'])

                # Display statistics
                st.success("Analysis complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(df))
                with col2:
                    st.metric("Unique Sizes", len(unique_sizes))
                with col3:
                    st.metric("Unique Formats", len(unique_formats))

                # Show sample images
                st.subheader("Sample Images")
                sample_size = min(5, len(df))
                sample_df = df.sample(sample_size)

                cols = st.columns(sample_size)
                for idx, (col, (_, row)) in enumerate(zip(cols, sample_df.iterrows())):
                    try:
                        img = Image.open(row['imagepath'])
                        col.image(img, caption=f"ID: {row['imageid']}\n{row['width']}x{row['height']}px\n{row['format']}",
                                width=150)
                    except Exception as e:
                        col.error(f"Couldn't display image: {str(e)}")

                # Show data table
                st.subheader("Image Metadata")
                st.dataframe(df)

                # Show visualizations
                st.subheader("Visualizations")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                # Size distribution
                df['resolution'] = df['width'].astype(
                    str) + 'x' + df['height'].astype(str)
                size_counts = df['resolution'].value_counts().head(10)
                size_counts.plot(kind='bar', ax=ax1, color='skyblue')
                ax1.set_title('Top 10 Image Resolutions')
                ax1.set_xlabel('Resolution (width x height)')
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)

                # Format distribution
                format_counts = df['format'].value_counts()
                format_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
                ax2.set_title('Image Format Distribution')
                ax2.set_ylabel('')

                st.pyplot(fig)

                # Add download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download metadata as CSV",
                    data=csv,
                    file_name='image_metadata.csv',
                    mime='text/csv',
                )
            else:
                st.warning("No valid images found in the directory!")


    def main():
        # App title and description
        st.title("Average Color Intensity Distribution in the Dataset")
        st.markdown("""
        This tool analyzes the color distribution across your uploaded images.
        The histogram shows the distribution of average red, green, and blue channel intensities.
        """)


    def process_image(image_file_name, image_directory):
        full_image_path = os.path.join(image_directory, image_file_name)
        with Image.open(full_image_path) as img:
            img_array = np.array(img)
            avg_color = np.mean(img_array, axis=(0, 1))
            return avg_color


def main():
    st.title("Average Color Intensity Distribution Analyzer")

    # File uploader for directory selection
    image_directory = st.text_input("Enter the path to your image directory:")
    pattern = re.compile(r'.*\.(jpg|jpeg|png)$', re.IGNORECASE)

    if image_directory and os.path.isdir(image_directory):
        # List of image filenames
        image_files = [f for f in os.listdir(
            image_directory) if pattern.match(f)]

        if not image_files:
            st.warning("No valid image files found in the directory!")
            return

        st.write(f"Found {len(image_files)} images to process...")

        # Process images with progress bar
        progress_bar = st.progress(0)
        avg_colors = []

        with ThreadPoolExecutor() as executor:
            # Create a list of tuples with directory path for each image
            tasks = [(img, image_directory) for img in image_files]
            for i, result in enumerate(executor.map(lambda x: process_image(*x), tasks)):
                avg_colors.append(result)
                progress_bar.progress((i + 1) / len(image_files))

        avg_colors = np.array(avg_colors)

        # Plot histograms
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(avg_colors[:, 0], bins=50,
                color='blue', alpha=0.6, label='Blue')
        ax.hist(avg_colors[:, 1], bins=50,
                color='green', alpha=0.6, label='Green')
        ax.hist(avg_colors[:, 2], bins=50, color='red', alpha=0.6, label='Red')
        ax.set_title("Average Color Intensity Distribution in the Dataset")
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Number of Images")
        ax.legend()

        st.pyplot(fig)

        # Show some statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Blue Intensity",
                      f"{np.mean(avg_colors[:, 0]):.1f}")
        with col2:
            st.metric("Mean Green Intensity",
                      f"{np.mean(avg_colors[:, 1]):.1f}")
        with col3:
            st.metric("Mean Red Intensity", f"{np.mean(avg_colors[:, 2]):.1f}")
    else:
        st.info("Please enter a valid directory path containing images.")


def compute_sharpness(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    return None


def process_image_sharpness(image_file, image_directory):
    image_path = os.path.join(image_directory, image_file)
    return compute_sharpness(image_path)


def main():
    st.title("Image Sharpness Distribution Analyzer")

    # File uploader for directory selection
    image_directory = st.text_input("Enter the path to your image directory:")
    pattern = re.compile(r'.*\.(jpg|jpeg|png)$', re.IGNORECASE)

    if image_directory and os.path.isdir(image_directory):
        # List of image filenames
        image_files = [f for f in os.listdir(
            image_directory) if pattern.match(f)]

        if not image_files:
            st.warning("No valid image files found in the directory!")
            return

        st.write(f"Found {len(image_files)} images to process...")

        # Process images with progress bar
        progress_bar = st.progress(0)
        sharpness_values = []

        with ThreadPoolExecutor() as executor:
            # Process images in parallel
            futures = []
            for img in image_files:
                futures.append(executor.submit(
                    process_image_sharpness, img, image_directory))

            for i, future in enumerate(futures):
                sharpness = future.result()
                if sharpness is not None:
                    sharpness_values.append(sharpness)
                progress_bar.progress((i + 1) / len(image_files))

        if not sharpness_values:
            st.error("No valid sharpness values could be computed!")
            return

        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sharpness_values, bins=50, color='blue', alpha=0.7)
        ax.set_title("Distribution of Image Sharpness in the Dataset")
        ax.set_xlabel("Laplacian Variance (Sharpness)")
        ax.set_ylabel("Number of Images")
        ax.grid(True, linestyle='--', alpha=0.7)

        st.pyplot(fig)

        # Show statistics
        st.subheader("Sharpness Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Sharpness", f"{np.mean(sharpness_values):.1f}")
        with col2:
            st.metric("Median Sharpness", f"{np.median(sharpness_values):.1f}")
        with col3:
            st.metric("Standard Deviation", f"{np.std(sharpness_values):.1f}")

        # Show distribution info
        st.subheader("Distribution Information")
        st.write(f"Minimum sharpness: {min(sharpness_values):.1f}")
        st.write(f"Maximum sharpness: {max(sharpness_values):.1f}")
        st.write(f"25th percentile: {np.percentile(sharpness_values, 25):.1f}")
        st.write(f"75th percentile: {np.percentile(sharpness_values, 75):.1f}")

    else:
        st.info("Please enter a valid directory path containing images.")

############


# Set page config
st.set_page_config(page_title="Image Sharpness Analyzer", layout="wide")

st.title("🔍 Image Sharpness Distribution Tool")
st.write("This app computes the Laplacian variance (sharpness) of uploaded images and displays their distribution.")

# Upload multiple images
uploaded_files = st.file_uploader("Upload multiple image files", type=[
                                  "jpg", "jpeg", "png"], accept_multiple_files=True)

# Function to compute sharpness


def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# Process images and compute sharpness
sharpness_values = []

if uploaded_files:
    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            sharpness = compute_sharpness(image)
            sharpness_values.append(sharpness)

    if sharpness_values:
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(sharpness_values, bins=50, color='blue', alpha=0.7)
        ax.set_title("Distribution of Image Sharpness in the Dataset")
        ax.set_xlabel("Laplacian Variance (Sharpness)")
        ax.set_ylabel("Number of Images")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.success(f"✅ Processed {len(sharpness_values)} images successfully.")
    else:
        st.warning("No valid images found.")
else:
    st.info("Please upload one or more images to analyze.")

##################################################


def main():
    st.title("Image Dataset Analysis")

    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Sharpness Distribution", "Color Distribution"])

    with tab1:
        st.header("Image Sharpness Distribution")

        # Generate mock data similar to your chart (replace with your actual data loading)
        # In a real app, you'd load your actual image files and compute sharpness
        np.random.seed(42)
        sharpness_values = np.random.gamma(2, 1000, 40000)
        sharpness_values = sharpness_values[sharpness_values < 40000]

        # Create the histogram plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sharpness_values, bins=50, color='blue', alpha=0.7)
        ax.set_title("Distribution of Image Sharpness in the Dataset")
        ax.set_xlabel("Laplacian Variance (Sharpness)")
        ax.set_ylabel("Number of Images")
        ax.grid(True, linestyle='--', alpha=0.7)

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Show statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Sharpness", f"{np.mean(sharpness_values):.2f}")
        col2.metric("Median Sharpness", f"{np.median(sharpness_values):.2f}")
        col3.metric("Total Images", len(sharpness_values))

    with tab2:
        st.header("Color Distribution Analysis")

        # Generate mock color data (replace with your actual data loading)
        np.random.seed(42)
        avg_colors = np.random.randint(0, 256, size=(10000, 3))

        # Create the color histogram plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(avg_colors[:, 0], bins=50,
                color='blue', alpha=0.6, label='Blue')
        ax.hist(avg_colors[:, 1], bins=50,
                color='green', alpha=0.6, label='Green')
        ax.hist(avg_colors[:, 2], bins=50, color='red', alpha=0.6, label='Red')
        ax.set_title("Average Color Intensity Distribution in the Dataset")
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Number of Images")
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)


def create_base64_thumbnail(image_path, size=(50, 50)):
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            encoded_img = base64.b64encode(
                img_byte_arr.getvalue()).decode('ascii')
            return f'data:image/jpeg;base64,{encoded_img}'
    except Exception as e:
        st.error(f"Error processing image {image_path}: {e}")
        return None


def main():
    st.title("Image Thumbnail Viewer")

    # File uploader for CSV or images
    uploaded_file = st.file_uploader(
        "Upload CSV file with image paths", type=['csv'])

    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)

            # Check if required columns exist
            if 'imagepath' not in df.columns:
                st.error("CSV file must contain an 'imagepath' column")
                return

            # Process images and create thumbnails
            st.info("Processing images...")
            df['thumbnail'] = df['imagepath'].apply(
                lambda x: create_base64_thumbnail(x))

            # Convert numeric columns if they exist
            if 'productid' in df.columns:
                df['productid'] = pd.to_numeric(
                    df['productid'], errors='coerce')
            if 'imageid' in df.columns:
                df['imageid'] = pd.to_numeric(df['imageid'], errors='coerce')

            # Display the first 40 images in a grid
            st.subheader("Image Thumbnails")
            cols = st.columns(5)  # 5 columns for the grid

            for idx, row in df.head(40).iterrows():
                with cols[idx % 5]:
                    if row['thumbnail']:
                        st.image(row['thumbnail'], width=100,
                                 caption=f"ID: {row.get('imageid', '')}")
                    else:
                        st.warning("No thumbnail")

            # Show dataframe info
            st.subheader("Dataset Information")
            st.dataframe(df.head())

            # Display statistics
            st.subheader("Statistics")
            col1, col2 = st.columns(2)
            col1.metric("Total Images", len(df))
            col2.metric("Missing Thumbnails", df['thumbnail'].isna().sum())

            # Show duplicates if 'productid' exists
            if 'productid' in df.columns:
                st.metric("Duplicate Product IDs", df.duplicated(
                    subset=['productid']).sum())

        except Exception as e:
            st.error(f"Error processing file: {e}")

#########


st.header("""Merging and sorting tables""")


def main():
    st.title("DataFrame Merging and Analysis")

    # File upload section
    st.sidebar.header("Upload Files")
    x_train_file = st.sidebar.file_uploader("Upload X_train CSV", type=['csv'])
    y_train_file = st.sidebar.file_uploader("Upload Y_train CSV", type=['csv'])
    df_images_file = st.sidebar.file_uploader(
        "Upload df_images CSV", type=['csv'])

    if x_train_file and y_train_file and df_images_file:
        try:
            # Load dataframes
            X_train = pd.read_csv(x_train_file)
            Y_train = pd.read_csv(y_train_file)
            df_images = pd.read_csv(df_images_file)

            # Show original data info
            st.subheader("Original Data Shapes")
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "X_train", f"{X_train.shape[0]} rows, {X_train.shape[1]} cols")
            col2.metric(
                "Y_train", f"{Y_train.shape[0]} rows, {Y_train.shape[1]} cols")
            col3.metric(
                "df_images", f"{df_images.shape[0]} rows, {df_images.shape[1]} cols")

            # Perform merges
            st.subheader("Merging DataFrames")

            with st.spinner("Merging X_train with Y_train..."):
                X_train = X_train.merge(Y_train, how='left', left_index=True, right_index=True,
                                        suffixes=('_X_train', '_Y_train'))
                st.success("First merge completed")

            with st.spinner("Merging with df_images..."):
                X_train = X_train.merge(df_images, on=['productid', 'imageid'], how='left',
                                        suffixes=('_X_train', '_df_images'))
                st.success("Second merge completed")

            # Column dropping
            cols_to_drop = ['Unnamed: 0_X_train', 'Unnamed: 0_Y_train', 'Unnamed: 0_df_images',
                            'prdtypecode_X_train', 'prdtypecode_Y_train', 'prdtypecode_df_images']

            cols_to_drop = [
                col for col in cols_to_drop if col in X_train.columns]

            if cols_to_drop:
                with st.spinner("Dropping columns..."):
                    X_train.drop(cols_to_drop, axis=1,
                                 inplace=True, errors='ignore')
                    st.success(f"Dropped columns: {', '.join(cols_to_drop)}")
            else:
                st.info("No columns to drop")

            # Display results
            st.subheader("Merged DataFrame")

            st.dataframe(X_train.head(10))

            st.subheader("Merged DataFrame Info")
            st.write(
                f"Shape: {X_train.shape[0]} rows, {X_train.shape[1]} columns")

            # Show column information
            if st.checkbox("Show column details"):
                st.write(X_train.dtypes)

            # Show statistics
            if st.checkbox("Show statistics"):
                st.write(X_train.describe())

            # Download option
            if st.button("Download Merged CSV"):
                csv = X_train.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Merged Data",
                    data=csv,
                    file_name="merged_data.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload all three CSV files to proceed")

#########


st.header("""modelling""")


#########
st.header("""image exploration""")

# Set page config
st.set_page_config(
    page_title="ML Image Analysis Suite",
    page_icon="📊",
    layout="wide"
)


def main():
    st.title("📊 ML Image Analysis Suite")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose functionality:",
                                ["Data Exploration", "Image Processing", "Model Training"])

    if app_mode == "Data Exploration":
        data_exploration()
    elif app_mode == "Image Processing":
        image_processing()
    elif app_mode == "Model Training":
        model_training()


def data_exploration():
    st.header("📈 Data Exploration")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success("Data loaded successfully!")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Basic stats
        if st.checkbox("Show basic statistics"):
            st.write(df.describe())

        # Column selector
        selected_col = st.selectbox("Select column for analysis", df.columns)

        # Visualization options
        viz_type = st.radio("Visualization type:",
                            ["Histogram", "Box Plot", "Count Plot"])

        if viz_type == "Histogram":
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], ax=ax)
            st.pyplot(fig)
        elif viz_type == "Box Plot":
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_col], ax=ax)
            st.pyplot(fig)
        elif viz_type == "Count Plot":
            fig, ax = plt.subplots()
            sns.countplot(x=df[selected_col], ax=ax)
            st.pyplot(fig)


def image_processing():
    st.header("🖼️ Image Processing")

    uploaded_images = st.file_uploader("Upload images",
                                       type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True)

    if uploaded_images:
        cols = st.columns(3)
        # Show max 9 images
        for idx, img_file in enumerate(uploaded_images[:9]):
            with cols[idx % 3]:
                img = Image.open(img_file)
                st.image(img, caption=img_file.name, width=200)

                # Process image on button click
                if st.button(f"Process {img_file.name}", key=f"btn_{idx}"):
                    with st.spinner("Processing..."):
                        # Convert to numpy array
                        img_array = np.array(img)

                        # Calculate sharpness
                        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

                        # Calculate average color
                        avg_color = np.mean(img_array, axis=(0, 1))

                        st.success(f"Sharpness: {sharpness:.2f}")
                        st.success(f"Average RGB: {avg_color}")


def model_training():
    st.header("🤖 Model Training")

    uploaded_file = st.file_uploader(
        "Upload training data (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data loaded successfully!")

        # Feature selection
        features = st.multiselect("Select features", df.columns)
        target = st.selectbox("Select target variable", df.columns)

        if features and target:
            X = df[features]
            y = df[target]

            # Test size slider
            test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2)

            if st.button("Train Model"):
                with st.spinner("Training in progress..."):
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    # Scale data
                    scaler = MinMaxScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train model
                    model = LinearRegression()
                    model.fit(X_train_scaled, y_train)

                    # Predictions
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_pred)

                    # Results
                    st.success("Training complete!")
                    st.metric("Mean Squared Error", f"{mse:.4f}")

                    # Plot results
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred)
                    ax.plot([y_test.min(), y_test.max()],
                            [y_test.min(), y_test.max()], 'k--', lw=2)
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    st.pyplot(fig)


#########
st.header("""image modeling""")

# Set page config
st.set_page_config(
    page_title="CNN Image Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

def build_model(input_shape, conv_filters, conv_k_reg, dense_k_reg, num_classes):
    """Build the CNN model architecture"""
    input_layer = Input(shape=input_shape)

    # Conv blocks
    x = Conv2D(filters=conv_filters[0], kernel_size=(3, 3),
               activation='relu', kernel_regularizer=l2(conv_k_reg))(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=conv_filters[0], kernel_size=(3, 3),
               activation='relu', kernel_regularizer=l2(conv_k_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Additional conv blocks would go here following the same pattern
    # ...

    x = GlobalAveragePooling2D()(x)

    # Dense layers
    x = Dense(512, activation='relu', kernel_regularizer=l2(dense_k_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(dense_k_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(dense_k_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer
    output = Dense(num_classes, activation='softmax',
                   kernel_regularizer=l2(0.001))(x)

    return Model(inputs=input_layer, outputs=output)


def main():
    st.title("📷 CNN Image Classification")

    # Sidebar for parameters
    with st.sidebar:
        st.header("Model Parameters")
        DATASET_PERC = st.slider("Dataset percentage", 0.1, 1.0, 0.9)
        IMG_SIZE = st.selectbox("Image size", [224, 256, 299], index=0)
        BATCH_SIZE = st.selectbox("Batch size", [32, 64, 128], index=1)
        N_EPOCHS = st.number_input("Number of epochs", 10, 200, 100)
        LR = st.number_input("Learning rate", 0.0001, 0.1,
                             0.01, step=0.001, format="%.4f")

        CONV_FILTERS = st.text_input(
            "Conv filters (comma separated)", "16,16,32,64,128")
        CONV_K_REG = st.number_input(
            "Conv kernel regularizer", 0.00001, 0.01, 0.0001, step=0.0001, format="%.5f")
        DENSE_K_REG = st.number_input(
            "Dense kernel regularizer", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")

        st.markdown("---")
        st.info("Adjust parameters and click 'Train Model' to start training.")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df_full = pd.read_csv(uploaded_file)
        df = df_full.head(int(df_full.shape[0]*DATASET_PERC))

        # Data preparation
        df['prdtypecode_str'] = df['prdtypecode'].astype(str)
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=66,
            stratify=df['prdtypecode'])

        # Compute class weights
        y_train_for_weights = train_df['prdtypecode'].astype(
            'category').cat.codes
        u_classes = np.unique(y_train_for_weights)
        class_weights = compute_class_weight(
            'balanced', classes=u_classes, y=y_train_for_weights)
        class_weight_dict = {int(cls): round(float(weight), 3)
                             for cls, weight in zip(u_classes, class_weights)}

        st.subheader("Class Distribution")
        st.write(pd.DataFrame.from_dict(class_weight_dict,
                 orient='index', columns=['Weight']))

        # Data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest')

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='imagepath',
            y_col='prdtypecode_str',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=66)

        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='imagepath',
            y_col='prdtypecode_str',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            seed=66)

        # Build and train model
        if st.button("Train Model"):
            with st.spinner("Building and training model..."):
                # Parse conv filters
                conv_filters = [int(x) for x in CONV_FILTERS.split(",")]

                # Build model
                model = build_model(
                    input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    conv_filters=conv_filters,
                    conv_k_reg=CONV_K_REG,
                    dense_k_reg=DENSE_K_REG,
                    num_classes=len(df['prdtypecode'].unique())
                )

                # Compile model
                f1_metric = F1Score(average='macro', name='f1_score')
                early_stopping = EarlyStopping(
                    monitor='val_f1_score', patience=7, restore_best_weights=True, mode='max')
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, mode='min')

                model.compile(
                    optimizer=AdamW(learning_rate=LR, weight_decay=0.01),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', f1_metric]
                )

                # Display model summary
                st.subheader("Model Summary")
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text("\n".join(model_summary))

                # Train model
                history = model.fit(
                    train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=N_EPOCHS,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    callbacks=[early_stopping, reduce_lr],
                    class_weight=class_weight_dict,
                    verbose=1
                )

                # Evaluation
                st.success("Training completed!")

                # Generate predictions
                val_generator.reset()
                y_pred = model.predict(val_generator)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true = val_generator.classes
                class_names = list(val_generator.class_indices.keys())

                # Generate reports
                class_report = classification_report(y_true, y_pred_classes,
                                                     target_names=class_names,
                                                     output_dict=True)

                # Display results
                st.subheader("Training Results")

                # Metrics columns
                col1, col2, col3 = st.columns(3)
                col1.metric("Final Training Accuracy",
                            f"{history.history['accuracy'][-1]:.4f}")
                col2.metric("Final Validation Accuracy",
                            f"{history.history['val_accuracy'][-1]:.4f}")
                col3.metric("Macro Avg F1",
                            f"{class_report['macro avg']['f1-score']:.4f}")

                # Training curves
                st.subheader("Training Curves")
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))

                # Loss
                axes[0, 0].plot(history.history['loss'], label='Train')
                axes[0, 0].plot(history.history['val_loss'],
                                label='Validation')
                axes[0, 0].set_title('Loss')
                axes[0, 0].legend()

                # Accuracy
                axes[0, 1].plot(history.history['accuracy'], label='Train')
                axes[0, 1].plot(history.history['val_accuracy'],
                                label='Validation')
                axes[0, 1].set_title('Accuracy')
                axes[0, 1].legend()

                # F1 Score
                axes[1, 0].plot(history.history['f1_score'], label='Train')
                axes[1, 0].plot(history.history['val_f1_score'],
                                label='Validation')
                axes[1, 0].set_title('F1 Score')
                axes[1, 0].legend()

                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred_classes)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
                axes[1, 1].set_title('Confusion Matrix')

                st.pyplot(fig)

                # Classification report
                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(class_report).transpose())


st.header("""Text exploration""")
# Set page config
st.set_page_config(
    page_title="CNN Image Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

def build_model(input_shape, conv_filters, conv_k_reg, dense_k_reg, num_classes):
    """Build the CNN model architecture"""
    input_layer = Input(shape=input_shape)

    # Conv blocks
    x = Conv2D(filters=conv_filters[0], kernel_size=(3, 3),
               activation='relu', kernel_regularizer=l2(conv_k_reg))(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=conv_filters[0], kernel_size=(3, 3),
               activation='relu', kernel_regularizer=l2(conv_k_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Additional conv blocks would go here following the same pattern
    # ...

    x = GlobalAveragePooling2D()(x)

    # Dense layers
    x = Dense(512, activation='relu', kernel_regularizer=l2(dense_k_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(dense_k_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(dense_k_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer
    output = Dense(num_classes, activation='softmax',
                   kernel_regularizer=l2(0.001))(x)

    return Model(inputs=input_layer, outputs=output)


def main():
    st.title("📷 CNN Image Classification")

    # Sidebar for parameters
    with st.sidebar:
        st.header("Model Parameters")
        DATASET_PERC = st.slider("Dataset percentage", 0.1, 1.0, 0.9)
        IMG_SIZE = st.selectbox("Image size", [224, 256, 299], index=0)
        BATCH_SIZE = st.selectbox("Batch size", [32, 64, 128], index=1)
        N_EPOCHS = st.number_input("Number of epochs", 10, 200, 100)
        LR = st.number_input("Learning rate", 0.0001, 0.1,
                             0.01, step=0.001, format="%.4f")

        CONV_FILTERS = st.text_input(
            "Conv filters (comma separated)", "16,16,32,64,128")
        CONV_K_REG = st.number_input(
            "Conv kernel regularizer", 0.00001, 0.01, 0.0001, step=0.0001, format="%.5f")
        DENSE_K_REG = st.number_input(
            "Dense kernel regularizer", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")

        st.markdown("---")
        st.info("Adjust parameters and click 'Train Model' to start training.")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df_full = pd.read_csv(uploaded_file)
        df = df_full.head(int(df_full.shape[0]*DATASET_PERC))

        # Data preparation
        df['prdtypecode_str'] = df['prdtypecode'].astype(str)
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=66,
            stratify=df['prdtypecode'])

        # Compute class weights
        y_train_for_weights = train_df['prdtypecode'].astype(
            'category').cat.codes
        u_classes = np.unique(y_train_for_weights)
        class_weights = compute_class_weight(
            'balanced', classes=u_classes, y=y_train_for_weights)
        class_weight_dict = {int(cls): round(float(weight), 3)
                             for cls, weight in zip(u_classes, class_weights)}

        st.subheader("Class Distribution")
        st.write(pd.DataFrame.from_dict(class_weight_dict,
                 orient='index', columns=['Weight']))

        # Data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest')

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='imagepath',
            y_col='prdtypecode_str',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=66)

        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='imagepath',
            y_col='prdtypecode_str',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            seed=66)

        # Build and train model
        if st.button("Train Model"):
            with st.spinner("Building and training model..."):
                # Parse conv filters
                conv_filters = [int(x) for x in CONV_FILTERS.split(",")]

                # Build model
                model = build_model(
                    input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    conv_filters=conv_filters,
                    conv_k_reg=CONV_K_REG,
                    dense_k_reg=DENSE_K_REG,
                    num_classes=len(df['prdtypecode'].unique())
                )

                # Compile model
                f1_metric = F1Score(average='macro', name='f1_score')
                early_stopping = EarlyStopping(
                    monitor='val_f1_score', patience=7, restore_best_weights=True, mode='max')
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, mode='min')

                model.compile(
                    optimizer=AdamW(learning_rate=LR, weight_decay=0.01),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', f1_metric]
                )

                # Display model summary
                st.subheader("Model Summary")
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text("\n".join(model_summary))

                # Train model
                history = model.fit(
                    train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=N_EPOCHS,
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    callbacks=[early_stopping, reduce_lr],
                    class_weight=class_weight_dict,
                    verbose=1
                )

                # Evaluation
                st.success("Training completed!")

                # Generate predictions
                val_generator.reset()
                y_pred = model.predict(val_generator)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true = val_generator.classes
                class_names = list(val_generator.class_indices.keys())

                # Generate reports
                class_report = classification_report(y_true, y_pred_classes,
                                                     target_names=class_names,
                                                     output_dict=True)

                # Display results
                st.subheader("Training Results")

                # Metrics columns
                col1, col2, col3 = st.columns(3)
                col1.metric("Final Training Accuracy",
                            f"{history.history['accuracy'][-1]:.4f}")
                col2.metric("Final Validation Accuracy",
                            f"{history.history['val_accuracy'][-1]:.4f}")
                col3.metric("Macro Avg F1",
                            f"{class_report['macro avg']['f1-score']:.4f}")

                # Training curves
                st.subheader("Training Curves")
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))

                # Loss
                axes[0, 0].plot(history.history['loss'], label='Train')
                axes[0, 0].plot(history.history['val_loss'],
                                label='Validation')
                axes[0, 0].set_title('Loss')
                axes[0, 0].legend()

                # Accuracy
                axes[0, 1].plot(history.history['accuracy'], label='Train')
                axes[0, 1].plot(history.history['val_accuracy'],
                                label='Validation')
                axes[0, 1].set_title('Accuracy')
                axes[0, 1].legend()

                # F1 Score
                axes[1, 0].plot(history.history['f1_score'], label='Train')
                axes[1, 0].plot(history.history['val_f1_score'],
                                label='Validation')
                axes[1, 0].set_title('F1 Score')
                axes[1, 0].legend()

                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred_classes)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
                axes[1, 1].set_title('Confusion Matrix')

                st.pyplot(fig)

                # Classification report
                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(class_report).transpose())


st.header("""vectorizing""")

# Set page config
st.set_page_config(
    page_title="NLP Text Processing",
    layout="wide",
    initial_sidebar_state="expanded"
)


class Word2VecEmbedder:
    '''takes a list of tokenized texts and returns their embeddings using Word2Vec'''

    def __init__(self, vector_size=300, min_count=2, workers=4):
        self.vector_size = vector_size
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, token_lists):
        st.info('Training Word2Vec...')
        self.model = Word2Vec(token_lists, vector_size=self.vector_size,
                              min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, token_lists):
        if self.model is None:
            raise ValueError('Model not trained. Call fit() first.')

        def text_to_vector(token_lists):
            vectors = [self.model.wv[word]
                       for word in token_lists if word in self.model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

        embeddings = np.array([text_to_vector(text) for text in token_lists])
        st.info(f'Word2Vec shape: {embeddings.shape}')
        return embeddings

    def fit_transform(self, token_lists):
        return self.fit(token_lists).transform(token_lists)


def main():
    st.title("📝 NLP Text Processing")

    # Sidebar for parameters
    with st.sidebar:
        st.header("Processing Parameters")
        DATASET_PERC = st.slider("Dataset percentage", 0.1, 1.0, 0.2)
        VOCAB_LIMIT = st.number_input("Vocabulary limit", 1000, 100000, 40000)
        MAX_SEQUENCE_LENGTH = st.number_input(
            "Max sequence length", 10, 200, 75)
        VECTOR_SIZE = st.number_input("TF-IDF vector size", 1000, 30000, 15000)

        st.markdown("---")
        st.info("Adjust parameters and click 'Process Data' to start analysis.")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Data loaded successfully! Shape: {df.shape}")

        # Sample data display
        if st.checkbox("Show sample data"):
            st.dataframe(df.head(3))

        # Load stopwords
        if st.checkbox("Load stopwords"):
            all_stopwords = np.load("all_stopwords.npy", allow_pickle=True)
            st.write("First 100 stopwords:", all_stopwords[:100])

        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                # Process subset of data
                df = df.head(int(df.shape[0]*DATASET_PERC))
                st.info(
                    f'Using {DATASET_PERC*100}% of the dataset for training ({len(df)} rows)')

                # TF-IDF Processing
                st.subheader("TF-IDF Vectorization")

                # Create tabs for different vectorization approaches
                tab1, tab2 = st.tabs(["Combined Text", "Preprocessed Tokens"])

                with tab1:
                    st.write("TF-IDF on combined text column")
                    vectorizer = TfidfVectorizer(
                        max_features=VECTOR_SIZE,
                        ngram_range=(1, 2),
                        min_df=5, max_df=0.8,
                        lowercase=True,
                        stop_words=list(all_stopwords),
                        sublinear_tf=True,
                        norm='l2')

                    tfidf_vectors = vectorizer.fit_transform(df['combined'])
                    st.write("TF-IDF vectors shape:", tfidf_vectors.shape)
                    st.write("Sample vectors:", tfidf_vectors[:3].toarray())

                with tab2:
                    st.write("TF-IDF on preprocessed tokens")
                    df['comb_tokens'] = df['comb_tokens'].apply(
                        ast.literal_eval)
                    vectorizer = TfidfVectorizer(
                        max_features=VECTOR_SIZE,
                        ngram_range=(1, 2),
                        min_df=5, max_df=0.8,
                        lowercase=True,
                        stop_words=list(all_stopwords),
                        sublinear_tf=True,
                        norm='l2')

                    # Convert tokens back to strings for TF-IDF
                    df['tokens_as_text'] = df['comb_tokens'].apply(
                        lambda x: ' '.join(x))
                    tfidf_tok_vectors = vectorizer.fit_transform(
                        df['tokens_as_text'])
                    st.write("TF-IDF tokens vectors shape:",
                             tfidf_tok_vectors.shape)
                    st.write("Sample vectors:",
                             tfidf_tok_vectors[:3].toarray())

                # Word2Vec Processing
                st.subheader("Word2Vec Embeddings")

                w2v = Word2VecEmbedder(vector_size=300, min_count=1)
                X_train_w2v = w2v.fit_transform(
                    df['comb_tokens'][:1000])  # First 1000 rows for demo

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Vocabulary size:", len(
                        w2v.model.wv.key_to_index))
                    st.write("Sample words:", list(
                        w2v.model.wv.key_to_index.keys())[:10])

                with col2:
                    sample_tokens = df['comb_tokens'].iloc[0]
                    st.write("Sample tokens:", sample_tokens)
                    embedding = w2v.transform([sample_tokens])
                    st.write("Embedding shape:", embedding.shape)
                    st.write("First 5 values:", embedding[0][:5])
                    st.write("Is all zeros?", np.all(embedding == 0))

                # Tokenization Comparison
                st.subheader("Tokenization Comparison")

                # Preprocess comb_tokens
                df['comb_tokens'] = df['comb_tokens'].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                all_tokens_filtered = []
                for token_list in df['comb_tokens']:
                    all_tokens_filtered.extend(token_list)

                # Count token frequencies
                token_counter = Counter(all_tokens_filtered)
                most_common_tokens = dict(
                    token_counter.most_common(VOCAB_LIMIT))
                vocab_filtered = {word: i+1 for i,
                                  word in enumerate(most_common_tokens.keys())}
                vocab_size_filtered = len(vocab_filtered) + 1

                def tokens_to_sequences_filtered(token_list):
                    return [vocab_filtered[token] for token in token_list if token in vocab_filtered]

                sequences = [tokens_to_sequences_filtered(
                    tokens) for tokens in df['comb_tokens']]
                X = pad_sequences(
                    sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

                st.write("Padded sequences shape:", X.shape)
                st.write("Sample sequence:", X[0])

                # Keras Tokenization
                st.write("Keras Tokenization:")

                def remove_stopwords(text):
                    words = str(text).lower().split()
                    return ' '.join([word for word in words if word not in all_stopwords])

                df['keras_tokens'] = df['combined'].apply(remove_stopwords)

                tokenizer = Tokenizer(num_words=VOCAB_LIMIT)
                tokenizer.fit_on_texts(df['keras_tokens'])
                sequences = tokenizer.texts_to_sequences(df['keras_tokens'])
                X_keras = pad_sequences(
                    sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

                st.write("Keras padded sequences shape:", X_keras.shape)
                st.write("Sample Keras sequence:", X_keras[0])

                # Tokenization Analysis
                st.subheader("Tokenization Analysis")

                # Display token frequency distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                top_tokens = token_counter.most_common(20)
                sns.barplot(x=[count for token, count in top_tokens],
                            y=[token for token, count in top_tokens], ax=ax)
                ax.set_title("Top 20 Most Frequent Tokens")
                st.pyplot(fig)


st.header("""Text modeling""")

# Set page config
st.set_page_config(page_title="Text Classification App", layout="wide")

# Title and description
st.title("Text Classification with Multiple Models")
st.write("""
This app allows you to train and evaluate different machine learning models for text classification.
You can choose between different text processing methods and model configurations.
""")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Load data


@st.cache_data
def load_data():
    DATA_PATH = '../data/'
    PROC_DATA_PATH = '../processed_data/'
    df = pd.read_parquet(PROC_DATA_PATH + 'X_train_with_labels_ext.parquet')
    all_stopwords = np.load(
        PROC_DATA_PATH + 'all_stopwords.npy', allow_pickle=True).tolist()
    return df, all_stopwords


try:
    df, all_stopwords = load_data()
    st.sidebar.success("Data loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    st.stop()

# Display data info
if st.sidebar.checkbox("Show data info"):
    st.subheader("Data Overview")
    st.write(f"Data shape: {df.shape}")
    st.write("First few rows:")
    st.dataframe(df.head())

# Constants
random_state = 66
VECTOR_SIZE = st.sidebar.slider("TF-IDF Vector Size", 5000, 30000, 15000, 1000)

# Model selection
model_options = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        solver='liblinear',
        random_state=random_state
    ),
    'SGD Classifier': SGDClassifier(
        loss='log_loss',
        alpha=0.0001,
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state
    ),
    'Linear SVM': LinearSVC(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
}

selected_models = st.sidebar.multiselect(
    "Select models to train",
    list(model_options.keys()),
    default=list(model_options.keys())
)

# Text processing options
text_option = st.sidebar.radio(
    "Select text processing method",
    ('Combined Text', 'Preprocessed Tokens')
)

# SMOTE option
use_smote = st.sidebar.checkbox("Use SMOTE for class balancing", value=True)

# Main function


def train_and_evaluate():
    st.subheader("Model Training and Evaluation")

    # Prepare data based on selection
    if text_option == 'Combined Text':
        X = df['combined']
    else:
        X = df['comb_tokens']

    y = df['prdtypecode']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Vectorize text
    vectorizer = TfidfVectorizer(
        max_features=VECTOR_SIZE,
        ngram_range=(1, 2),
        min_df=5, max_df=0.8,
        lowercase=True,
        stop_words=all_stopwords,
        sublinear_tf=True,
        norm='l2'
    )

    st.write(f"Vectorizing text using {text_option}...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    st.write(f"TF-IDF matrix shape: {X_train_vec.shape}")

    # Train and evaluate models
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, model_name in enumerate(selected_models):
        status_text.text(f"Training {model_name}...")
        progress_bar.progress((i + 1) / len(selected_models))

        model = model_options[model_name]

        # Encode labels
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_test_enc = label_encoder.transform(y_test)

        # Apply SMOTE if selected
        if use_smote:
            smote = SMOTE(random_state=random_state,
                          k_neighbors=5, sampling_strategy='auto')
            X_train_res, y_train_res = smote.fit_resample(
                X_train_vec, y_train_enc)
            model.set_params(class_weight=None)
        else:
            X_train_res, y_train_res = X_train_vec, y_train_enc

        # Train model
        model.fit(X_train_res, y_train_res)

        # Predict
        y_pred_enc = model.predict(X_test_vec)
        y_pred = label_encoder.inverse_transform(y_pred_enc)

        # Calculate metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')

        results.append({
            'Model': model_name,
            'F1 Macro': f1_macro,
            'F1 Weighted': f1_weighted
        })

    # Display results
    results_df = pd.DataFrame(results)
    st.subheader("Model Performance")
    st.dataframe(results_df.style.highlight_max(axis=0))

    # Show classification report for the best model
    best_model_idx = results_df['F1 Macro'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_model = model_options[best_model_name]

    if use_smote:
        best_model.fit(X_train_res, y_train_res)
    else:
        best_model.fit(X_train_vec, y_train_enc)

    y_pred = best_model.predict(X_test_vec)
    y_pred = label_encoder.inverse_transform(y_pred)

    st.subheader(f"Classification Report for {best_model_name}")
    st.text(classification_report(y_test, y_pred))


# Run the training
if st.sidebar.button("Train Models"):
    train_and_evaluate()

# Add some visualizations
if st.sidebar.checkbox("Show class distribution"):
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    df['prdtypecode'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
Created with Streamlit  
Text Classification App  
""")

if __name__ == "__main__":
    main()
