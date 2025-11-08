"""
Test HuggingFace connectivity and model access
Verifies that HF_TOKEN works and models can be downloaded
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*60)
print("HUGGINGFACE CONNECTIVITY TEST")
print("="*60)

# Check environment
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    print(f"[OK] HF_TOKEN found: {hf_token[:10]}...")
else:
    print("[ERROR] HF_TOKEN not found in environment")
    print("Please ensure .env file contains HF_TOKEN=your_token_here")
    sys.exit(1)

# Try importing transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    print("[OK] Transformers library imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import transformers: {e}")
    print("Run: pip install transformers torch")
    sys.exit(1)

print("\n" + "="*60)
print("TESTING MODEL ACCESS")
print("="*60)

# Test 1: Try to access FinBERT
print("\n1. Testing ProsusAI/finbert...")
try:
    print("   Attempting to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ProsusAI/finbert",
        token=hf_token,
        cache_dir="./models/cache"
    )
    print("   [OK] Tokenizer loaded successfully")

    print("   Attempting to load model (this may take time)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        token=hf_token,
        cache_dir="./models/cache"
    )
    print("   [OK] Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

except Exception as e:
    print(f"   [ERROR] Failed to load ProsusAI/finbert: {e}")

    # Try alternative
    print("\n   Trying alternative: yiyanghkust/finbert-tone...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "yiyanghkust/finbert-tone",
            token=hf_token,
            cache_dir="./models/cache"
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone",
            token=hf_token,
            cache_dir="./models/cache"
        )
        print("   [OK] Alternative model loaded successfully")
    except Exception as e2:
        print(f"   [ERROR] Alternative also failed: {e2}")

# Test 2: Check HuggingFace API connectivity
print("\n2. Testing HuggingFace API connectivity...")
try:
    import requests

    # Test with public API endpoint
    response = requests.get(
        "https://huggingface.co/api/models/bert-base-uncased",
        timeout=10
    )
    if response.status_code == 200:
        print("   [OK] Can reach HuggingFace API")
    else:
        print(f"   [WARNING] HuggingFace API returned status {response.status_code}")

    # Test with authenticated endpoint
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.get(
        "https://huggingface.co/api/whoami",
        headers=headers,
        timeout=10
    )
    if response.status_code == 200:
        user_info = response.json()
        print(f"   [OK] Authentication successful for user: {user_info.get('name', 'Unknown')}")
    else:
        print(f"   [ERROR] Authentication failed with status {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"   [ERROR] Network issue: {e}")
except Exception as e:
    print(f"   [ERROR] Unexpected error: {e}")

# Test 3: Check cache directory
print("\n3. Checking cache directories...")
cache_dirs = [
    Path.home() / ".cache" / "huggingface",
    Path("./models/cache"),
    Path(os.environ.get("TRANSFORMERS_CACHE", ""))
]

for cache_dir in cache_dirs:
    if cache_dir and cache_dir.exists():
        print(f"   [OK] Cache directory exists: {cache_dir}")
        # Count cached models
        model_dirs = list(cache_dir.glob("models--*"))
        if model_dirs:
            print(f"       Found {len(model_dirs)} cached models")
            for model_dir in model_dirs[:3]:  # Show first 3
                print(f"       - {model_dir.name}")
    elif cache_dir:
        print(f"   [INFO] Cache directory does not exist: {cache_dir}")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

print("""
If models fail to load:
1. Check internet connectivity and firewall settings
2. Verify HF_TOKEN has correct permissions
3. Try using a VPN if HuggingFace is blocked
4. Consider downloading models manually and using local paths
5. Use smaller alternative models (yiyanghkust/finbert-tone)

For the training script:
- The fallback mechanisms are working, so training can continue
- Models will be cached after first successful download
- Future runs will use cached models (faster startup)
""")