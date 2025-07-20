import streamlit as st
import os

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

prdtypes_en = {
 10: "Used books",
 40: "Video games",
 50: "Video game accessories",
 60: "Video game consoles",
 1140: "Children's figurines",
 1160: "Trading cards",
 1180: "Adult figurines and role-playing games",
 1280: "Toys",
 1281: "Board games",
 1300: "Remote-controlled toys",
 1301: "Baby socks",
 1302: "Children's fishing",
 1320: "Childcare",
 1560: "Interior Furniture",
 1920: "Bedding",
 1940: "Food",
 2060: "Decoration",
 2220: "Pets",
 2280: "Magazines",
 2403: "Magazines, Books and Comics",
 2462: "Used Games",
 2522: "Stationery",
 2582: "Garden Furniture",
 2583: "Swimming Pool Equipment",
 2585: "Maintenance",
 2705: "New Books",
 2905: "PC Games"
}

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

def select_h5_file(folder_path="."):
    """
    Creates a Streamlit selectbox to choose an .h5 file from a specified folder.

    Args:
        folder_path (str): The path to the folder to search for .h5 files.
                           Defaults to the current directory.
    Returns:
        str: The selected .h5 file name, or None if no files are found or selected.
    """
    h5_files = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".h5") and os.path.isfile(os.path.join(folder_path, file)):
                h5_files.append(file.replace('.h5', ''))
        h5_files.sort() # Sort alphabetically for better user experience
    else:
        st.warning(f"Folder not found: {folder_path}. Please ensure the folder exists.")
        return None

    if not h5_files:
        st.info(f"No .h5 files found in '{folder_path}'.")
        return None
    else:
        selected_file = st.selectbox(
            "Select an .h5 model file:",
            options=["Select"] + h5_files, # Add an empty option to allow no selection initially
            index=0, # Default to the empty option
            help="Choose a Keras/TensorFlow model file (e.g., .h5, .keras)."
        )
        return selected_file if selected_file else None

