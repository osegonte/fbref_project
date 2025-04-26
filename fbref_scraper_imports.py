import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import random
import psycopg2
from psycopg2.extras import execute_values
import io
import hashlib
from dotenv import load_dotenv
from rate_limit_handler import RateLimitHandler

# Load environment variables
load_dotenv()

# Create directories if they don't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("data/cache", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/fbref_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fbref_scraper")