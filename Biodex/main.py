import os
import time
import random
import json
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

# Set page config immediately
st.set_page_config(
    page_title="BioDex - The Pocket Zoologist AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LangChain and Agent Libraries
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Vision and Image Processing Libraries
from transformers import pipeline
import torch
from PIL import Image

# Web and Mapping Libraries
import requests
from bs4 import BeautifulSoup
import folium
from streamlit_folium import folium_static  # using st_folium via folium_static

# Additional Research Toolkits from Phidata
from phi.tools.pubmed import PubmedTools
# (Arxiv toolkit removed per requirements)

# ---------------------------------------
# Load Environment Variables
# ---------------------------------------
load_dotenv()

# ---------------------------------------
# Inject Tailwind CSS via CDN
# ---------------------------------------
import streamlit as st
import streamlit_authenticator as stauth
import pickle

# Load the user configuration
with open('config.pkl', 'rb') as file:
    config = pickle.load(file)

# Initialize the authenticator
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    cookie_key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days'],
    preauthorized=config['preauthorized']
)

# Display the login widget
authenticator.login(location='main')

# Handle authentication status
if st.session_state["authentication_status"]:
    authenticator.logout('Logout', location='sidebar')
    st.sidebar.success(f"🔓 Logged in as {st.session_state['name']}")
    # Proceed with the rest of your app
elif st.session_state["authentication_status"] is False:
    st.error("❌ Incorrect username or password")
    st.stop()
elif st.session_state["authentication_status"] is None:
    st.warning("🛡 Please enter your login credentials to continue")
    st.stop()



st.markdown("""
    <style>
        @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Initialize DuckDuckGo Search Tool (for general facts)
# ---------------------------------------
search_tool = DuckDuckGoSearchRun()

# ---------------------------------------
# Vision Model: Load ViT-B/16 for Animal Recognition
# ---------------------------------------
import os
import streamlit as st
from transformers import pipeline, AutoModelForImageClassification, AutoProcessor

# List of valid categories for animals and organisms
VALID_CATEGORIES = {
    'animal', 'bird', 'mammal', 'reptile', 'amphibian', 'fish', 'insect', 'arachnid',
    'crustacean', 'mollusk', 'worm', 'microorganism', 'bacteria', 'fungus', 'plant',
    'vertebrate', 'invertebrate', 'wildlife', 'fauna', 'organism'
}

# Comprehensive list of known species organized by category
KNOWN_SPECIES = {
    # Birds
    'macaw', 'parrot', 'cockatoo', 'toucan', 'lorikeet', 'eagle', 'hawk', 'owl', 'penguin',
    'sparrow', 'finch', 'canary', 'budgie', 'lovebird', 'cockatiel', 'conure', 'quaker',
    'pigeon', 'dove', 'chicken', 'duck', 'goose', 'swan', 'turkey', 'peacock', 'pheasant',
    'ostrich', 'emu', 'kiwi', 'flamingo', 'pelican', 'stork', 'heron', 'egret', 'crane',
    'seagull', 'albatross', 'vulture', 'falcon', 'kite', 'buzzard', 'woodpecker', 'kingfisher',
    'hummingbird', 'swallow', 'martin', 'warbler', 'thrush', 'robin', 'cardinal', 'bluejay',
    'crow', 'raven', 'magpie', 'starling', 'mynah', 'oriole', 'tanager', 'grosbeak',
    
    # Mammals
    'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'monkey', 'gorilla', 'chimpanzee',
    'orangutan', 'lemur', 'sloth', 'anteater', 'armadillo', 'pangolin', 'hedgehog',
    'rabbit', 'hare', 'squirrel', 'chipmunk', 'beaver', 'porcupine', 'raccoon', 'skunk',
    'fox', 'wolf', 'coyote', 'jackal', 'hyena', 'bear', 'panda', 'polar bear', 'grizzly',
    'weasel', 'otter', 'badger', 'wolverine', 'ferret', 'mink', 'seal', 'sea lion',
    'walrus', 'dolphin', 'whale', 'orca', 'narwhal', 'beluga', 'manatee', 'dugong',
    'bat', 'kangaroo', 'koala', 'wombat', 'platypus', 'echidna', 'deer', 'elk', 'moose',
    'antelope', 'gazelle', 'buffalo', 'bison', 'yak', 'camel', 'llama', 'alpaca',
    'rhinoceros', 'hippopotamus', 'tapir', 'pig', 'boar', 'cow', 'bull', 'sheep', 'goat',
    
    # Reptiles
    'snake', 'python', 'boa', 'cobra', 'viper', 'rattlesnake', 'lizard', 'gecko', 'chameleon',
    'iguana', 'monitor', 'komodo', 'turtle', 'tortoise', 'terrapin', 'crocodile', 'alligator',
    'caiman', 'gavial',
    
    # Amphibians
    'frog', 'toad', 'salamander', 'newt', 'caecilian',
    
    # Fish
    'shark', 'ray', 'stingray', 'eel', 'salmon', 'trout', 'tuna', 'bass', 'perch', 'pike',
    'catfish', 'goldfish', 'koi', 'angelfish', 'tetra', 'guppy', 'molly', 'swordtail',
    'betta', 'clownfish', 'tang', 'wrasse', 'grouper', 'snapper', 'cod', 'halibut',
    'flounder', 'sole', 'mackerel', 'herring', 'sardine', 'anchovy',
    
    # Insects
    'butterfly', 'moth', 'dragonfly', 'damselfly', 'grasshopper', 'cricket', 'katydid',
    'beetle', 'ladybug', 'firefly', 'ant', 'termite', 'bee', 'wasp', 'hornet', 'yellowjacket',
    'mosquito', 'fly', 'housefly', 'fruit fly', 'dragonfly', 'damselfly', 'mantis',
    'stick insect', 'walking stick', 'cockroach', 'earwig', 'silverfish', 'centipede',
    'millipede',
    
    # Arachnids
    'spider', 'tarantula', 'scorpion', 'tick', 'mite',
    
    # Crustaceans
    'crab', 'lobster', 'shrimp', 'prawn', 'crayfish', 'krill', 'barnacle',
    
    # Mollusks
    'octopus', 'squid', 'cuttlefish', 'nautilus', 'snail', 'slug', 'clam', 'oyster',
    'mussel', 'scallop',
    
    # Marine Invertebrates
    'jellyfish', 'coral', 'anemone', 'sea urchin', 'starfish', 'sea cucumber', 'sponge',
    
    # Microorganisms
    'bacteria', 'virus', 'fungus', 'mold', 'yeast', 'algae', 'protozoa', 'amoeba',
    'paramecium', 'euglena',
    
    # Plants (basic categories)
    'tree', 'flower', 'grass', 'fern', 'moss', 'lichen', 'algae', 'fungus', 'mushroom'
}

def is_valid_organism(label: str, confidence: float) -> bool:
    """
    Validate if the detected object is an animal or organism.
    Returns True if the label contains any valid category or known species and confidence is high enough.
    """
    label_lower = label.lower()
    
    # Check if any valid category is in the label
    is_valid_category = any(category in label_lower for category in VALID_CATEGORIES)
    
    # Check if any known species is in the label
    is_known_species = any(species in label_lower for species in KNOWN_SPECIES)
    
    # If it's a known species or valid category, use a lower threshold
    if is_valid_category or is_known_species:
        return confidence >= 0.6  # Lower threshold for known species/categories
    
    # For unknown species, require higher confidence
    return confidence >= 0.9

@st.cache_resource
def load_vision_model():
    """Load the ViT model from the locally saved directory."""
    local_model_path = "./models/vit_model"  # Path where you saved the model

    return pipeline(
        task="image-classification",
        model=AutoModelForImageClassification.from_pretrained(local_model_path),
        feature_extractor=AutoProcessor.from_pretrained(local_model_path)
    )

def vision_agent(image):
    """Analyze the image using the cached vision model and validate results."""
    model_pipeline = load_vision_model()
    results = model_pipeline(image)
    
    # Filter and validate results
    valid_results = []
    for result in results:
        if is_valid_organism(result['label'], result['score']):
            valid_results.append(result)
    
    return valid_results if valid_results else results  # Return original results if no valid matches found

# ---------------------------------------
# Helper: Minimize Uploaded Image Size
# ---------------------------------------
def minimize_image_size(image: Image.Image, max_size: tuple = (1024, 1024)) -> Image.Image:
    """Resize the image to a maximum size to speed up processing."""
    image.thumbnail(max_size)
    return image

# ---------------------------------------
# Geographical Data and Mapping (GBIF)
# ---------------------------------------
def get_geographical_data(animal_name: str) -> list:
    """Fetch geographical distribution data using the GBIF API."""
    try:
        url = f"https://api.gbif.org/v1/occurrence/search?q={animal_name}&limit=100"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            locations = []
            for record in data.get("results", []):
                lat = record.get("decimalLatitude")
                lon = record.get("decimalLongitude")
                if lat is not None and lon is not None:
                    locations.append((lat, lon))
            return locations
        return []
    except Exception as e:
        st.error(f"Error fetching geographical data: {str(e)}")
        return []

def create_map(locations: list, animal_name: str) -> folium.Map:
    """Create an interactive map with markers for each location."""
    if not locations:
        return None
    m = folium.Map(location=locations[0], zoom_start=4)
    for lat, lon in locations:
        folium.Marker(
            location=[lat, lon],
            popup=f"{animal_name} Habitat",
            icon=folium.Icon(color="green")
        ).add_to(m)
    return m

# ---------------------------------------
# PubMed Data Functions (for Genetic Data and Biological Research)
# ---------------------------------------
def get_pubmed_genetic_data(animal_name: str) -> str:
    """Fetch genetic data for the given organism from PubMed."""
    try:
        toolkit = PubmedTools(email=os.getenv("PUBMED_EMAIL", "your_email@example.com"))
        query = f"genetic data {animal_name}"
        result = toolkit.search_pubmed(query)
        return result
    except Exception as e:
        return f"Error fetching genetic data from PubMed: {str(e)}"

def get_pubmed_bio_research(animal_name: str) -> str:
    """Fetch biological research articles for the given organism from PubMed."""
    try:
        toolkit = PubmedTools(email=os.getenv("PUBMED_EMAIL", "your_email@example.com"))
        query = f"biological research {animal_name}"
        result = toolkit.search_pubmed(query)
        return result
    except Exception as e:
        return f"Error fetching biological research from PubMed: {str(e)}"

# ---------------------------------------
# Helper: Format PubMed Output as Bullet Points
# ---------------------------------------
def format_pubmed_info(pubmed_text: str) -> str:
    """
    Attempt to format PubMed output as bullet points.
    If the text is JSON, parse it; otherwise, split by newlines.
    """
    try:
        data = json.loads(pubmed_text)
        lines = []
        if isinstance(data, list):
            for article in data:
                if isinstance(article, dict):
                    title = article.get("Title", "No Title")
                    published = article.get("Published", "Unknown Date")
                    summary = article.get("Summary", "No summary available")
                    lines.append(f"- *{title}* (Published: {published}): {summary}")
                else:
                    lines.append(f"- {article}")
            return "\n".join(lines)
        else:
            return pubmed_text
    except Exception:
        lines = pubmed_text.split("\n")
        bullet_lines = [f"- {line.strip()}" for line in lines if line.strip()]
        return "\n".join(bullet_lines)

# ---------------------------------------
# LangChain Tools Setup
# ---------------------------------------
tools = [
    Tool(
        name="General Info",
        func=search_tool.run,
        description="Fetch general facts and articles about the given organism using DuckDuckGo.",
    ),
    Tool(
        name="Genetic Data",
        func=get_pubmed_genetic_data,
        description="Fetch genetic data for the given organism from PubMed.",
    ),
    Tool(
        name="Geographical Data",
        func=get_geographical_data,
        description="Fetch geographical distribution data for the given organism using the GBIF API.",
    ),
    Tool(
        name="Biological Research",
        func=get_pubmed_bio_research,
        description="Fetch biological research articles related to the given organism from PubMed.",
    ),
]

# Initialize the LangChain Agent using ChatGroq
groq_model = ChatGroq(model="llama3-8b-8192")
agent = initialize_agent(
    tools=tools,
    llm=groq_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# ---------------------------------------
# Streamlit User Interface
# ---------------------------------------
st.title("🐾 BioDex - The Pocket Zoologist AI")

st.sidebar.markdown("""
    <div class="p-4">
        <h2 class="text-xl font-bold">About:</h2>
        <p>This app analyzes an uploaded animal image using a vision model,
           then fetches biological data and research articles via various APIs.</p>
    </div>
""", unsafe_allow_html=True)

# File uploader for the image with Tailwind styling
# File uploader for the image with Tailwind styling
uploaded_file = st.file_uploader("Upload an image of an animal:", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # 👇 NEW: Show original info before shrinking
    st.write("📷 **Before Shrinking**")
    st.write(f"Original size (pixels): {image.size[0]} x {image.size[1]}")
    st.write(f"Original pixel count: {image.size[0] * image.size[1]:,}")

    image = minimize_image_size(image)  # Resize image for faster processing

    # 👇 NEW: Show shrunk info after shrinking
    st.write("📉 **After Shrinking**")
    st.write(f"Shrunk size (pixels): {image.size[0]} x {image.size[1]}")
    st.write(f"Shrunk pixel count: {image.size[0] * image.size[1]:,}")

    # 👇 NEW: Calculate and show reduction %
    reduction = (1 - (image.size[0] * image.size[1]) /
                   (uploaded_file.size if hasattr(uploaded_file, "size") else (image.size[0] * image.size[1]))) * 100
    st.write(f"⚡ Approx reduction in workload: {reduction:.2f}%")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    if st.button("🔍 Analyze Image", key="analyze"):
        with st.spinner("Analyzing image..."):
            vision_results = vision_agent(image)
            st.json(vision_results)
            
            if vision_results and isinstance(vision_results, list) and len(vision_results) > 0:
                most_likely_animal = vision_results[0]['label']
                confidence_score = vision_results[0]['score']
                
                # Check if the result is a valid organism
                if is_valid_organism(most_likely_animal, confidence_score):
                    st.subheader(f"Most Likely Organism Detected: *{most_likely_animal}* (Confidence: {confidence_score:.2f})")
                    
                    if confidence_score > 0.7:  # Increased confidence threshold
                        with st.spinner(f"Fetching data for {most_likely_animal}..."):
                            # 1. General Facts and Articles (trimmed to 300 characters)
                            try:
                                general_info = search_tool.run(most_likely_animal)
                                if len(general_info) > 300:
                                    general_info = general_info[:300] + "..."
                            except Exception as e:
                                st.error(f"Error fetching general info: {e}")
                                general_info = ""
                            
                            # 2. Genetic Data via PubMed (formatted as bullet points)
                            genetic_data = get_pubmed_genetic_data(most_likely_animal)
                            formatted_genetic = format_pubmed_info(genetic_data)
                            
                            # 3. Geographical Distribution via GBIF
                            locations = get_geographical_data(most_likely_animal)
                            map_obj = create_map(locations, most_likely_animal) if locations else None
                            
                            # 4. Biological Research via PubMed (formatted as bullet points)
                            bio_research = get_pubmed_bio_research(most_likely_animal)
                            formatted_bio = format_pubmed_info(bio_research)
                            
                        st.markdown('<div class="mt-6 p-6 bg-gray-100 rounded-lg shadow-lg">', unsafe_allow_html=True)
                        st.markdown("## Organism Analysis Report", unsafe_allow_html=True)
                        
                        if general_info.strip():
                            st.markdown('<div class="mb-4 p-4 bg-white rounded shadow">', unsafe_allow_html=True)
                            st.markdown("### 1. General Facts and Articles", unsafe_allow_html=True)
                            st.write(general_info)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        if formatted_genetic.strip() and "error fetching" not in formatted_genetic.lower():
                            st.markdown('<div class="mb-4 p-4 bg-white rounded shadow">', unsafe_allow_html=True)
                            st.markdown("### 2. Genetic Data (PubMed)", unsafe_allow_html=True)
                            st.markdown(formatted_genetic, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        if map_obj:
                            st.markdown('<div class="mb-4 p-4 bg-white rounded shadow">', unsafe_allow_html=True)
                            st.markdown("### 3. Geographical Distribution", unsafe_allow_html=True)
                            folium_static(map_obj)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        if formatted_bio.strip() and "error fetching" not in formatted_bio.lower():
                            st.markdown('<div class="mb-4 p-4 bg-white rounded shadow">', unsafe_allow_html=True)
                            st.markdown("### 4. Biological Research (PubMed)", unsafe_allow_html=True)
                            st.markdown(formatted_bio, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("No animal or organism detected in the image. Please upload an image containing an animal, bird, or other organism.")
            else:
                st.error("No animal or organism detected in the image. Please upload an image containing an animal, bird, or other organism.")

st.markdown("""
    <footer class="text-center p-4 mt-8 bg-blue-500 text-white rounded">
      <p>Built with ❤️ using AI and Streamlit</p>
    </footer>
""", unsafe_allow_html=True)