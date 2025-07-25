import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import Counter
from nltk.corpus import stopwords
from pathlib import Path
from wordcloud import WordCloud
import io
import json
import keras
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import models
from common import custom_stopwords, display_paired_images_in_reports_folder, get_image_full_path, prdtypes, prdtypes_en, select_h5_file, word_grouping
from image_preprocessing import (load_original_image, get_random_image_path, baseline_preprocessing, 
        advanced_augmentation_preprocessing, background_removal_preprocessing, smart_crop_preprocessing)
from functions import display_html_file, show_pdf_page, load_keras_model_and_predict, numpy_to_json

import warnings
warnings.filterwarnings('ignore')

DATA_RAW = './data/raw'
DATA_PROCESSED = './data/processed'
DIR_MARKDOWN = './markdown'
DIR_HTML = './html'
DIR_MODELS = './models'
DIR_SAMPLE_IMAGES = './sample_images'
img_size = 224

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

# --- PAGE CONFIG MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="Rakuten E-commerce Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading (Cached for efficiency) ---
@st.cache_data
def load_all_data():
    try:
        X_test_df = pd.read_parquet(f"{DATA_RAW}/X_test_update.parquet")
        X_train_df = pd.read_parquet(f"{DATA_RAW}/X_train_update.parquet")
        X_train_ready = pd.read_parquet(f"{DATA_PROCESSED}/X_train_ready.parquet")
        Y_train_df = pd.read_parquet(f"{DATA_RAW}/Y_train_CVw08PX.parquet")
        # Merge Y_train into X_train immediately after loading to ensure consistency across reruns
        X_train_df = X_train_df.merge(Y_train_df, how='left', left_index=True,
                                      right_index=True, suffixes=('_X_train', '_Y_train'))
        return X_test_df, X_train_df, Y_train_df, X_train_ready
    except FileNotFoundError as e:
        st.error(
            f"Error loading data: {e}. Please ensure data files are in the '{DATA_RAW}' directory.")
        st.stop()  # Stop the app if data is not found
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        st.stop()

# --- NLTK Downloads (Cached for efficiency) ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        nltk.download('stopwords')
    try:
        # Required for word_tokenize, though not explicitly used here, good to have if needed later
        nltk.data.find('tokenizers/punkt')
    except Exception:
        nltk.download('punkt')

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
    st.pyplot(fig)

# Frequency tables
def get_freq_df(tokens):
    return pd.DataFrame(Counter(tokens).most_common(20), columns=["Word", "Frequency"])

download_nltk_data()
# Stopwords and replacements
french_stopwords = set(stopwords.words('french'))
all_stopwords = french_stopwords.union(custom_stopwords)
X_test, X_train, Y_train, X_train_ready = load_all_data()

text_model = "text-cnn-epochs-100-lr-0.001-testing.keras"
text_history = "text-cnn-epochs-100-lr-0.001-testing_history.json"
text_report = 'training_text-cnn-epochs-100-lr-0.001-testing_f1-0.7257_t-0721_1642.pdf'
#text_predictions = ''

multimodal_model = "multimodal-mobilenetv2--lr-0.0001_f1-0.795.keras"
multimodal_history = "multimodal-mobilenetv2--lr-0.0001_f1-0.795_history.json"
multimodal_report = 'training_multimodal-mobilenetv2--lr-0.0001_f1-0.795_f1-0.7950_t-0722_2033.pdf'
multimodal_predictions = 'multimodal-mobilenetv2--lr-0.0001_f1-0.795_predictions.npy'

# --- PAGE CONFIG MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="Rakuten E-commerce Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Streamlit specific code ----
st.title("Rakuten e-commerce project")
st.sidebar.title("Table of contents")
pages = [
    "Introduction",
    "Data Processing",
    "Data Exploration",
    "Image Processing",
    "Modelling",
    "Interpretation",
    "Prediction",
    "Conclusions"
]
page = st.sidebar.radio("Go to", pages)
page_current = 0

# --- Page 0: Introduction ---------------------------------------------------
if page == pages[page_current]:
    st.title("Introduction")
    # Page configuration
    st.set_page_config(page_title="Rakuten Product Classification", layout="centered")
    st.title("Classification of Rakuten E-Commerce Products")

    introduction = read_markdown_file(f"{DIR_MARKDOWN}/introduction.md")
    st.markdown(introduction, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.caption("Developed as part of the Rakuten France Multimodal Product Classification Challenge.")


# --- Page 1: Data Processing ---
page_current = page_current + 1
if page == pages[page_current]:
    st.subheader("Presentation of DataFrames")

    tables = [('X_train', X_train), ('X_test', X_test), ('Y_train', Y_train)]
    for table_name, table in tables:
        st.markdown(f"\n**{table_name}**\n")
        st.dataframe(table.head())

        buffer = io.StringIO()
        table.info(buf=buffer)
        s = buffer.getvalue()
        # st.text(s)
        # st.write(f"Duplicates: {table.duplicated().sum()}")

    # st.subheader("The training data (merged with Y_train)")
    # st.dataframe(X_train.head())
    # st.write(f"Shape of X_train: {X_train.shape}")
    # st.subheader("The dataset description")
    # st.dataframe(X_train.describe())
    # if st.checkbox("Show NA counts for X_train"):
    #     st.dataframe(X_train.isna().sum())

    st.markdown("""
    In each column, we are going to investigate:
    1. Missing values
    2. Duplicates
    3. Unique modalities
    """)

    missing_x = X_train.isnull().sum()
    missing_x_percent = X_train.isnull().mean() * 100
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
    duplicate_imageid_percent = X_train.duplicated(
        subset=["imageid"]).mean() * 100
    duplicate_prdtypecode_percent = X_train.duplicated(
        subset=["prdtypecode"]).mean() * 100

    unique_designation = X_train['designation'].nunique()
    unique_description = X_train['description'].nunique()
    unique_productids = X_train['productid'].nunique()
    unique_imageids = X_train['imageid'].nunique()
    unique_prdtypecode = X_train['prdtypecode'].nunique()

    # Creation of a MultiIndex for the "Check column"
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
        # Removed [:15] as index_tuples list has exactly 15 elements now
    }, index=index)

    st.title("Data Quality Checks")

    st.dataframe(check_df.style.format(
        # Added % sign to format
        {"Values (%)": "{:.2f}%"}), use_container_width=True)

    st.markdown(
        f"**The exact number of duplicates by line is:** {X_train.duplicated().sum()}")

    st.header("First Analysis")
    analysis_1 = read_markdown_file(f"{DIR_MARKDOWN}/analysis_1.md")
    st.markdown(analysis_1, unsafe_allow_html=True)


# --- Page 2: DataVizualization ---
page_current = page_current + 1
if page == pages[page_current]:
    st.header("Product type identification")
    data = X_train.copy()
    # st.dataframe(data.head())

    st.title("Word Clouds and Frequency Tables by Product Type")
    st.subheader(
        "(Type Name assumed based on observations in French & English)")
    #         "Choose a product type:", sorted(data['prdtypecode'].unique()))
    selected_type = st.selectbox(
        "Choose a product type:", prdtypes,
        format_func=lambda x: f"(prdtypecode: {str(x)}) {prdtypes.get(x)} [{prdtypes_en.get(x)}]")

    subset = data[data['prdtypecode'] == selected_type]
    designation_tokens = subset['designation'].dropna().astype(
        str).apply(clean_text).sum()
    description_tokens = subset['description'].dropna().astype(
        str).apply(clean_text).sum()
    # combined_tokens = designation_tokens + description_tokens

    subset_ready = X_train_ready[X_train_ready['prdtypecode'] == selected_type]
    combined_tokens = subset_ready['comb_tokens_fr'].astype(
        str).apply(clean_text).sum()

    # Layout for word clouds
    st.subheader("Word Clouds")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_wordcloud(designation_tokens, "Designation", "Reds")
    with col2:
        plot_wordcloud(description_tokens, "Description", "Blues")
    with col3:
        plot_wordcloud(combined_tokens, "Combined", "Greens")

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

    # Add the mapping as a new column (X_train is already merged)
    # Y_train["prdtype"] = Y_train["prdtypecode"].map(prdtypes) # Y_train is not used later in this block for display, only X_train
    X_train["prdtype"] = X_train["prdtypecode"].map(prdtypes)

    # Show a sample table
    st.subheader("Sample of Mapped Product Types in X_train")
    st.dataframe(
        X_train.sample(20)[['prdtypecode', 'prdtype',]].sort_values(by="prdtypecode"),
        use_container_width=True,
        # hide_index=True
    )

    st.subheader("Distribution of products across product types")
    prdtypecode_count = X_train['prdtypecode'].value_counts()

    # Key statistics
    prdtypecode_max_frequency = prdtypecode_count.idxmax()
    prdtypecode_min_frequency = prdtypecode_count.idxmin()
    prdtypecode_avg_frequency = round(prdtypecode_count.mean(), 0)
    prdtypecode_med_frequency = round(prdtypecode_count.median(), 0)
    prdtypecode_std_classes = round(prdtypecode_count.std(), 0)
    imbalance_ratio = round(prdtypecode_count.max() /
                            prdtypecode_count.min(), 1)

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

    st.subheader("General statistics")
    st.dataframe(stat_analysis, use_container_width=True)

    st.write("Distribution")
    prdtypecode_count_index = prdtypecode_count.index
    prdtypecode_proportions = prdtypecode_count / len(X_train) * 100
    prdtype_sorted_list = [prdtypes[code] for code in prdtypecode_count_index]

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

    # st.header("Distribution of products per product Type Code")
    # prdtypecode_count_df = pd.DataFrame({
    #     "Product Type": prdtype_sorted_list,
    #     "Occurences": prdtypecode_count
    # })
    # st.dataframe(prdtypecode_count_df)

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
    X_train['designation_length'] = X_train['designation'].astype(
        str).str.len()
    X_train['description_length'] = X_train['description'].astype(
        str).str.len()

    # Group by 'prdtypecode' and count the null values in 'description' and total values
    null_counts = X_train.groupby('prdtypecode').agg({
        'description': lambda x: x.isnull().sum(),
        'designation': 'count'  # This counts non-null designations
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

    final_df = null_counts[['prdtypecode', 'prdtype',
                            'null_descriptions_count', 'null_descriptions_pct']]

    # Sort the DataFrame by the number of null values in 'description' in descending order
    final_df = final_df.sort_values('null_descriptions_count', ascending=False)

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

    # final_df_sorted = final_df.sort_values(
    #     'null_descriptions_pct', ascending=False)
    # st.dataframe(final_df_sorted)

    st.write('''We notice that 3 product types have >40% of null descriptions. 3 of these categories relate to books or magazines,
             suggesting that the title was a sufficient source of information for sellers and buyers, especially when it is second hand.
             We will have to take this in consideration in the future to prevent underfitting.''')

    st.write("Replicate descriptions")
    # Create a new column that checks if designation is identical to description
    X_train['identical_designation_description'] = X_train['designation'].astype(
        str) == X_train['description'].astype(str)
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
    identical_counts = identical_counts.rename(columns={
        'identical_designation_description': 'identical_count',
        'designation': 'total_count'
    })

    final_df = identical_counts[['prdtypecode',
                                 'prdtype', 'identical_count', 'identical_pct']]
    final_df = final_df.sort_values('identical_count', ascending=False)

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

    # final_df_sorted = final_df.sort_values('identical_pct', ascending=False)
    # st.dataframe(final_df_sorted)

    # Display sample of products with identical designation and description
    replicate_products = X_train[X_train['identical_designation_description']]
    replicate_products_count = replicate_products.shape[0]

    st.header(
        "Random Sample of 10 Products with Identical Designation and Description")
    st.write(
        f'There are {replicate_products_count} replicate products in total.')
    st.dataframe(replicate_products[[
        'prdtypecode', 'prdtype', 'designation', 'description']].sample(10))

    st.write('''Replicate dimensions is a limited phenomenon. It reaches 1.4% of products for Puériculture products (child care).
             We will still have this in mind for the preprocessing.''')

    # X_train['designation_length'] = X_train['designation'].str.len()
    # X_train['description_length'] = X_train['description'].str.len()
    
    # st.header("Duplicate designations and descriptions")
    # st.title("Duplicate Values Analysis")
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
    duplicate_counts['prdtype'] = duplicate_counts['prdtypecode'].map(prdtypes)

    # Calculate percentages
    duplicate_counts['duplicate_designations_pct'] = round(
        (duplicate_counts['designation_duplicates'] / duplicate_counts['designation_non_null_count']) * 100, 1)
    duplicate_counts['duplicate_descriptions_pct'] = round(
        (duplicate_counts['description_duplicates'] / duplicate_counts['description_non_null_count']) * 100, 1)

    duplicate_counts = duplicate_counts.rename(columns={
        'designation_duplicates': 'duplicate_designations_count',
        'description_duplicates': 'duplicate_descriptions_count',
    })
    final_df = duplicate_counts[['prdtypecode', 'prdtype', 'duplicate_designations_count',
                                 'duplicate_designations_pct', 'duplicate_descriptions_count',
                                 'duplicate_descriptions_pct']]

    final_df = final_df.sort_values(
        'duplicate_designations_count', ascending=False)
    mean_designation_count = final_df['duplicate_designations_count'].mean()
    mean_designation_pct = final_df['duplicate_designations_pct'].mean()
    mean_description_count = final_df['duplicate_descriptions_count'].mean()
    mean_description_pct = final_df['duplicate_descriptions_pct'].mean()

    # VISUALIZATION 1: DESIGNATION DUPLICATES
    # st.header(
    #     "Number of Duplicate Values in Designation per Product Type (Excluding Nulls)")
    # fig1, ax1 = plt.subplots(figsize=(14, 6))

    # sns.barplot(data=final_df, x='prdtype', y='duplicate_designations_count',
    #             ax=ax1, color='skyblue')
    # ax1.set_xlabel('Product Type')
    # ax1.set_ylabel('Number of Duplicate Values')
    # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right')
    # ax1.grid(True, linestyle=':', alpha=0.7)

    # Mean line
    # ax1.axhline(mean_designation_count, color='skyblue',
    #             linestyle=':', linewidth=1)
    # ax1.text(x=len(final_df)-0.5, y=mean_designation_count,
    #          s=f'Mean: {mean_designation_count:.1f}', color='skyblue',
    #          ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7))

    # Percentage line
    # ax2 = ax1.twinx()
    # sns.lineplot(data=final_df, x='prdtype', y='duplicate_designations_pct',
    #              ax=ax2, color='lightcoral', marker='o')
    # ax2.set_ylabel('Percentage of Duplicate Values (%)')
    # ax2.axhline(mean_designation_pct, color='lightcoral',
    #             linestyle=':', linewidth=1)
    # ax2.text(x=len(final_df)-0.5, y=mean_designation_pct,
    #          s=f'Mean (%): {mean_designation_pct:.1f}%', color='lightcoral',
    #          ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7))

    # st.pyplot(fig1)

    # # VISUALIZATION 2: DESCRIPTION DUPLICATES
    # st.header(
    #     "Number of Duplicate Values in Description per Product Type (Excluding Nulls)")
    # fig2, ax3 = plt.subplots(figsize=(14, 6))

    # sns.barplot(data=final_df, x='prdtype', y='duplicate_descriptions_count',
    #             ax=ax3, color='skyblue')
    # ax3.set_xlabel('Product Type')
    # ax3.set_ylabel('Number of Duplicate Values')
    # ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90, ha='right')
    # ax3.grid(True, linestyle=':', alpha=0.7)

    # # Mean line
    # ax3.axhline(mean_description_count, color='skyblue',
    #             linestyle=':', linewidth=1)
    # ax3.text(x=len(final_df)-0.5, y=mean_description_count,
    #          s=f'Mean: {mean_description_count:.1f}', color='skyblue',
    #          ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7))

    # # Percentage line
    # ax4 = ax3.twinx()
    # sns.lineplot(data=final_df, x='prdtype', y='duplicate_descriptions_pct',
    #              ax=ax4, color='lightcoral', marker='o')
    # ax4.set_ylabel('Percentage of Duplicate Values (%)')
    # ax4.axhline(mean_description_pct, color='lightcoral',
    #             linestyle=':', linewidth=1)
    # ax4.text(x=len(final_df)-0.5, y=mean_description_pct,
    #          s=f'Mean (%): {mean_description_pct:.1f}%', color='lightcoral',
    #          ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7))

    # st.pyplot(fig2)

    # # DATA TABLE
    # st.header("Duplicate Values Summary")
    # final_df_sorted = final_df.sort_values(
    #     'duplicate_descriptions_pct', ascending=False)
    # st.dataframe(final_df_sorted)

    # TITLE AND HEADER
    st.title("Duplicates Analysis")
    st.header(
        "Product Types by Percentage of Duplicate Designations vs. Descriptions")

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

    plt.title(
        'Product Types by Percentage of Duplicate Designations vs. Descriptions')
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

    plt.legend(
        title='Product Type',
        bbox_to_anchor=(0.5, -0.15),
        loc='upper center',
        ncol=3
    )
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    st.pyplot(fig)

    # Display final dataframe
    # st.header(
    #     "Number of Duplicate Values in Designation and Description per Product Type (Excluding Nulls)")
    # st.dataframe(final_df)
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
        top_designations['prdtype'] = top_designations['prdtypecode'].map(
            prdtypes)
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
        top_descriptions['prdtype'] = top_descriptions['prdtypecode'].map(
            prdtypes)
        st.dataframe(
            top_descriptions.sort_values('duplicate_count', ascending=False)
            # Completed the line
            .head(20)[['prdtypecode', 'prdtype', 'description', 'duplicate_count']]
        )
    
    summary_images = read_markdown_file(f"{DIR_MARKDOWN}/summary_images.md")
    st.markdown(summary_images, unsafe_allow_html=True)

# -----------------------------------------------------
page_current = page_current + 1
if page == pages[page_current]:
    st.title("🖼️ Image Preprocessing Methods Demo")
    st.markdown("### 📊 Basic Image Properties")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Images", "84,916")
        st.metric("Missing Images", "0")
        st.metric("Average Sharpness", "671.93")
        st.metric("Sharp Images (%)", "86%")

    with col2:
        st.metric("Average Brightness", "43.47")
        st.metric("Bright Images (%)", "35%")
        st.metric("Average Contrast", "62.93")


    st.markdown("*Compare 4 different image preprocessing approaches*")

    # Sidebar controls
    st.sidebar.header("Controls")

    # Random image button
    if st.sidebar.button("🎲 Select Random Image", type="primary"):
        st.session_state.current_image = get_random_image_path()
        st.session_state.processed_images = None

    # Initialize session state
    if 'current_image' not in st.session_state:
        st.session_state.current_image = get_random_image_path()
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = None

    # Display current image info
    st.info(f"**Current Image:** {os.path.basename(st.session_state.current_image)}")

    # Show original image continuously at the top
    if st.session_state.current_image:
        original = load_original_image(st.session_state.current_image)
        if original is not None:
            st.subheader("📸 Original Image")
            
            # Make the original image bigger by using columns
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(original, caption=f"Current: {os.path.basename(st.session_state.current_image)}", use_container_width=True)

    # Method descriptions - always visible as dropdown
    st.markdown("---")
    st.subheader("🔍 Method Descriptions")

    with st.expander("📖 Click to learn about each method"):

        preproc_img = read_markdown_file(f"{DIR_MARKDOWN}/image_preproc.md")
        st.markdown(preproc_img, unsafe_allow_html=True)

    # Show process instruction if not processed yet
    if not st.session_state.processed_images:
        st.info("👆 Click 'Process Image' to see the preprocessing results!")

    # Process images button
    if st.sidebar.button("🔄 Process Image", type="secondary"):
        with st.spinner("Processing image with all methods... This may take a moment."):
            
            if original is not None:
                # Process with all methods
                baseline = baseline_preprocessing(st.session_state.current_image)
                background_removed = background_removal_preprocessing(st.session_state.current_image)
                smart_crop = smart_crop_preprocessing(st.session_state.current_image)
                advanced = advanced_augmentation_preprocessing(st.session_state.current_image)
                
                # Store in session state
                st.session_state.processed_images = {
                    'baseline': baseline,
                    'background_removed': background_removed,
                    'smart_crop': smart_crop,
                    'advanced': advanced
                }
                
                st.success("✅ Image processed successfully!")
            else:
                st.error("Failed to load the selected image.")

    # Display results
    if st.session_state.processed_images:
        st.markdown("---")
        st.subheader("📊 Processing Results")
        
        # Create 4 columns for the processed images
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        columns = [col1, col2, col3, col4]
        methods = [
            ('baseline', '⚡ Baseline'),
            ('background_removed', '🎭 Background Removed'),
            ('smart_crop', '✂️ Smart Crop'),
            ('advanced', '🌟 Advanced Augmentation')
        ]
        
        for i, (method, title) in enumerate(methods):
            with columns[i]:
                st.markdown(f"**{title}**")
                
                img = st.session_state.processed_images[method]
                if img is not None:
                    # Convert to display format if needed
                    if img.dtype == np.float32:
                        display_img = (img * 255).astype(np.uint8)
                    else:
                        display_img = img
                    
                    st.image(display_img, use_container_width=True)
                    
                    # Show image stats
                    st.caption(f"Shape: {img.shape}")
                    if img.dtype == np.float32:
                        st.caption(f"Range: [{img.min():.3f}, {img.max():.3f}]")
                    else:
                        st.caption(f"Range: [{img.min()}, {img.max()}]")
                else:
                    st.error("Failed to process with this method")

    else:
        st.markdown("---")

# ---  Modeling ---------------------------
page_current = page_current + 1
if page == pages[page_current]:

    st.header("Modeling")
    # Model selection dropdown
    model_options = [
        "Select a model",
        "Basic ML Models", 
        "Custom CNN Text",
        "Multimodal Fusion Model"
    ]
    selected_model = st.selectbox("Choose a model:", model_options)

    if selected_model == "Basic ML Models":
        st.subheader("Basic ML Models")
        display_html_file(DIR_HTML +'/basic_models.html')
        
    elif selected_model == "Custom CNN Text":
        st.subheader("Custom CNN Text Model")

        display_html_file(DIR_HTML +'/text_model.html')

        # Load specific model files
        model_file = text_model
        history_file = text_history
        
        model_path = DIR_MODELS + f"/{model_file}"
        history_path = DIR_MODELS + f"/{history_file}"
        
        st.info(f"Loading model: {model_file}")
        st.info(f"Loading history: {history_file}")
        
        # Load Keras model
        try:
            model = keras.models.load_model(model_path)
            st.success(f"Complete Keras model loaded from: {model_path}")
        except Exception as e:
            st.error(f"Error loading Keras model: {e}")
            st.stop()
        
        # Load history from JSON
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
            st.success(f"Training history loaded from {history_path}")
            
            # Display model info
            model_name = "Custom CNN Text Model"
            st.title(f"CNN Text Model Results: {model_name}")
            
            # Create visualization
            fig = plt.figure(figsize=(12, 8))
            
            # Loss curve
            plt.subplot(2, 2, 1)
            plt.plot(history_data['loss'], label='Train Loss', linewidth=2)
            plt.plot(history_data['val_loss'], label='Val Loss', linewidth=2)
            plt.title('Training Loss', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # F1 Score curve (main focus)
            plt.subplot(2, 2, 2)
            if 'f1_score' in history_data:
                plt.plot(history_data['f1_score'], label='Train F1 Macro', linewidth=2)
                plt.plot(history_data['val_f1_score'], label='Val F1 Macro', linewidth=2)
                plt.title('F1 Score (Macro) - Primary Metric', fontweight='bold')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'F1 Macro not tracked', ha='center', va='center')
                plt.title('F1 Score (N/A)')
            plt.grid(True, alpha=0.3)
            
            # Learning Rate (if available)
            plt.subplot(2, 2, 3)
            if 'learning_rate' in history_data:
                plt.plot(history_data['learning_rate'], linewidth=2, color='red')
                plt.title('Learning Rate', fontweight='bold')
                plt.yscale('log')
            else:
                plt.text(0.5, 0.5, 'LR not tracked', ha='center', va='center')
                plt.title('Learning Rate (N/A)')
            plt.grid(True, alpha=0.3)
            
            # Model summary info
            plt.subplot(2, 2, 4)
            plt.axis('off')
            
            final_val_f1 = history_data.get('val_f1_score', ['N/A'])[-1] if 'val_f1_score' in history_data else 'N/A'
            final_train_f1 = history_data.get('f1_score', ['N/A'])[-1] if 'f1_score' in history_data else 'N/A'
            final_val_loss = history_data['val_loss'][-1]
            epochs = len(history_data['loss'])
            
            summary_text = f"""MODEL SUMMARY

            Epochs: {epochs}
            Final Train F1 Macro: {final_train_f1 if final_train_f1 != 'N/A' else 'N/A'}
            Final Val F1 Macro: {final_val_f1 if final_val_f1 != 'N/A' else 'N/A'}
            Final Val Loss: {final_val_loss:.3f}

            Parameters: {model.count_params():,}
            Architecture: Custom CNN Text
            """
            
            plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f'Training Results: {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show detailed metrics
            st.subheader("Training History Details")
            with st.expander("View Raw Training History"):
                st.json(history_data)
                
        else:
            st.error(f"Training history file not found: {history_path}")

        pdf_path = DIR_MODELS + f"/{text_report}"
        show_pdf_page(pdf_path, pnum=2)
            
    elif selected_model == "Multimodal Fusion Model":
        st.subheader("Multimodal Fusion Model")
        display_html_file(DIR_HTML +'/multimodal.html')
        display_html_file(DIR_HTML +'/technic.html')
        # Load specific multimodal model files
        model_file = multimodal_model
        history_file = multimodal_history

        model_path = DIR_MODELS + f"/{model_file}"
        history_path = DIR_MODELS +f"/{history_file}"

        st.info(f"Loading model: {model_file}")
        st.info(f"Loading history: {history_file}")
        
        # Load Keras model
        try:
            model = keras.models.load_model(model_path)
            st.success(f"Complete Keras model loaded from: {model_path}")
        except Exception as e:
            st.error(f"Error loading Keras model: {e}")
            st.stop()
        
        # Load history from JSON
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)
            st.success(f"Training history loaded from {history_path}")
            
            # Display model info
            model_name = "Multimodal Fusion Model (Text + MobileNetV2)"
            st.title(f"Multimodal Model Results: {model_name}")
            
            # Create visualization
            fig = plt.figure(figsize=(12, 8))
            
            # Loss curve
            plt.subplot(2, 2, 1)
            plt.plot(history_data['loss'], label='Train Loss', linewidth=2)
            plt.plot(history_data['val_loss'], label='Val Loss', linewidth=2)
            plt.title('Training Loss', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # F1 Score curve (main focus)
            plt.subplot(2, 2, 2)
            if 'f1_score' in history_data:
                plt.plot(history_data['f1_score'], label='Train F1 Macro', linewidth=2)
                plt.plot(history_data['val_f1_score'], label='Val F1 Macro', linewidth=2)
                plt.title('F1 Score (Macro) - Primary Metric', fontweight='bold')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'F1 Macro not tracked', ha='center', va='center')
                plt.title('F1 Score (N/A)')
            plt.grid(True, alpha=0.3)
            
            # Learning Rate (if available)
            plt.subplot(2, 2, 3)
            if 'learning_rate' in history_data:
                plt.plot(history_data['learning_rate'], linewidth=2, color='red')
                plt.title('Learning Rate', fontweight='bold')
                plt.yscale('log')
            else:
                plt.text(0.5, 0.5, 'LR not tracked', ha='center', va='center')
                plt.title('Learning Rate (N/A)')
            plt.grid(True, alpha=0.3)
            
            # Model summary info
            plt.subplot(2, 2, 4)
            plt.axis('off')
            
            final_val_f1 = history_data.get('val_f1_score', ['N/A'])[-1] if 'val_f1_score' in history_data else 'N/A'
            final_train_f1 = history_data.get('f1_score', ['N/A'])[-1] if 'f1_score' in history_data else 'N/A'
            final_val_loss = history_data['val_loss'][-1]
            epochs = len(history_data['loss'])
            
            summary_text = f"""MODEL SUMMARY

            Epochs: {epochs}
            Final Train F1 Macro: {final_train_f1 if final_train_f1 != 'N/A' else 'N/A'}
            Final Val F1 Macro: {final_val_f1 if final_val_f1 != 'N/A' else 'N/A'}
            Final Val Loss: {final_val_loss:.3f}

            Parameters: {model.count_params():,}
            Architecture: Multimodal (Text CNN + MobileNetV2)
            """
            
            plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f'Training Results: {model_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show detailed metrics
            st.subheader("Training History Details")
            with st.expander("View Raw Training History"):
                st.json(history_data)
                
        else:
            st.error(f"Training history file not found: {history_path}")

        pdf_path = DIR_MODELS+ f'/{multimodal_report}'
        show_pdf_page(pdf_path, pnum=2)
        
    else:
        st.write("Please select a model from the dropdown above.")

# --- Page 4: Interpretation ---
page_current = page_current + 1
if page == pages[page_current]:
    st.header("Interpretation Grad-CAM")
    display_paired_images_in_reports_folder("./reports/figures")

# --- Page 5: Prediction ---------------------------------------------------
page_current = page_current + 1
if page == pages[page_current]:
    st.title("Prediction")

    # Sidebar controls
    st.sidebar.header("Controls")
    X_test_short = X_test.sort_values(by="imageid", ascending=False).head(50)
    
    # Random image button
    if st.sidebar.button("🎲 Select Random Test Product", type="primary"):
        st.session_state.current_product = X_test_short.sample(n=1)
        st.session_state.processed_product = None

    # Initialize session state
    if 'current_product' not in st.session_state:
        st.session_state.current_product = X_test_short.sample(n=1)
    if 'processed_product' not in st.session_state:
        st.session_state.processed_product = None

    curr_product = st.session_state.current_product.to_dict('records')[0]
    curr_product_img = get_image_full_path(curr_product['imageid'], curr_product['productid'], './data/raw/images/images_test')
    # st.table(st.session_state.current_product)

    # Show original image continuously at the top
    if curr_product_img:
        original = load_original_image(curr_product_img)
        if original is not None:
            st.subheader(curr_product['designation'])
            st.text(curr_product['description'])
            
            # Make the original image bigger by using columns
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(original, caption=f"Current: {os.path.basename(curr_product_img)}", use_container_width=True)

    # Method descriptions - always visible as dropdown
    st.markdown("---")
    st.subheader("🔍 Predictions")
    
    df = st.session_state.current_product
    df['combined'] = df['designation'] + df['description']
    df['comb_tokens_fr'] = df['combined'].dropna().astype(str).apply(clean_text)
    # st.table(df)
         
    model_options = [
        "Select a model",
        "Custom CNN Text",
        "Multimodal Fusion Model"
    ]
    selected_model_predict = st.selectbox("Choose a model:", model_options)

    if selected_model_predict == "Custom CNN Text":
        st.subheader("Custom CNN Text Model")

        # Load specific model files
        model_file = text_model
        history_file = text_history        
        model_path = DIR_MODELS + f"/{model_file}"
        history_path = DIR_MODELS + f"/{history_file}"

        df_input = [df]
            
    elif selected_model_predict == "Multimodal Fusion Model":
        st.subheader("Multimodal Fusion Model")
        # Load specific multimodal model files
        model_file = multimodal_model
        history_file = multimodal_history
        model_path = DIR_MODELS + f"/{model_file}"
        history_path = DIR_MODELS +f"/{history_file}"

        img = load_img(curr_product_img, target_size=(img_size, img_size))
        img_array = img_to_array(img)
        df_input = [df, img_array]
        
    else:
        st.write("Please select a model from the dropdown above.")

    if selected_model_predict != 'Select a model':
        try:
            model = keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading Keras model: {e}")
            st.stop()
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history_data = json.load(f)                
        else:
            st.error(f"Training history file not found: {history_path}")

        vectors = model.layers[-1].trainable_weights[0].numpy()
        # print(vectors)
        st.write(f"Shape of loaded embedding vectors: {vectors.shape}")

        # If your model has multiple inputs:
        if isinstance(model.input, list):
            print("\nModel has multiple inputs:")
            for i, inp in enumerate(model.input):
                print(f"  Input {i+1} Expected Shape: {inp.shape}")
                expected_shape = inp.shape
                print(f"Model's Expected Input Shape: {expected_shape}")
                expected_dtype = inp.dtype
                print(f"Model's Expected Input Data Type: {expected_dtype}")
                if isinstance(model.input, list):
                    print("\nModel has multiple inputs:")
                    for i, inp in enumerate(model.input):
                        print(f"  Input {i+1} - Shape: {inp.shape}, Dtype: {inp.dtype}")


        predictions = load_keras_model_and_predict(model_path, df_input)
        # st.text(predictions)
        if predictions is not None:
            if len(predictions) > 0:
                st.json(numpy_to_json(predictions), expanded=False)

                # label_encoder = LabelEncoder()
                # st.write(vectors)
                
                predictions_output = np.array(predictions)
                # Get the index of the highest probability for the first (and only) sample
                predicted_class_index = np.argmax(predictions_output[0])

                # Get the confidence for that class
                confidence = predictions_output[0][predicted_class_index]

                st.text(f"Predicted Class Index (0-indexed): {predicted_class_index}")
                x = list(prdtypes)[predicted_class_index]
                st.text(f"(prdtypecode: {str(x)}) {prdtypes.get(x)} [{prdtypes_en.get(x)}]")
                st.text(f"Confidence (Probability): {confidence:.4f}")


# --- Page 6: Conclusion ---------------------------------------------------
page_current = page_current + 1
if page == pages[page_current]:
    st.title("Conclusion")
    conclusion = read_markdown_file(f"{DIR_MARKDOWN}/conclusion.md")
    st.markdown(conclusion, unsafe_allow_html=True)
