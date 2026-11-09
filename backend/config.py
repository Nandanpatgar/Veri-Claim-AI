import os
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# OLLAMA CONFIGURATION
# ==========================================

OLLAMA_BASE_URL = "http://localhost:11434"

OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

CLAIM_EXTRACTION_MODEL = "llama3.2"

OLLAMA_TIMEOUT = 120


# ==========================================
# WIKIPEDIA CONFIGURATION
# ==========================================

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

MAX_WIKIPEDIA_RESULTS = 5


# ==========================================
# TAVILY CONFIGURATION
# ==========================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

TAVILY_API_URL = "https://api.tavily.com/search"

MAX_TAVILY_RESULTS = 5


# ==========================================
# EVIDENCE RETRIEVAL CONFIGURATION
# ==========================================

MAX_EVIDENCE_PER_CLAIM = 10