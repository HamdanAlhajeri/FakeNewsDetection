"""
Quick connectivity test for the Tinker API.
Run from the project root: python src/tinker_test.py

Tests (in order):
  1. API key is present in .env
  2. Tinker server is reachable (health check)
  3. Authentication works (list jobs endpoint)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

TINKER_API_KEY = os.getenv('TINKER_API_KEY')
TINKER_API_URL = os.getenv('TINKER_API_URL', 'https://api.tinker.thinkingmachines.ai').rstrip('/')

print("=" * 55)
print("Tinker API Connection Test")
print("=" * 55)

# ── 1. Check key is present ──────────────────────────────
print("\n[1] Checking API key...")
if not TINKER_API_KEY or TINKER_API_KEY == 'your_api_key_here':
    print("    FAIL — TINKER_API_KEY not set in .env")
    sys.exit(1)
print(f"    OK  — key found ({TINKER_API_KEY[:8]}...)")

# ── 2. Health check ──────────────────────────────────────
print(f"\n[2] Reaching {TINKER_API_URL}/v1/health ...")
try:
    import requests
    resp = requests.get(f'{TINKER_API_URL}/v1/health', timeout=10)
    print(f"    HTTP {resp.status_code}")
    if resp.status_code == 200:
        print("    OK  — server is up")
    else:
        print(f"    WARN — unexpected status: {resp.text[:200]}")
except requests.exceptions.ConnectionError:
    print("    FAIL — could not reach server (check internet / API URL)")
    sys.exit(1)
except requests.exceptions.Timeout:
    print("    FAIL — request timed out")
    sys.exit(1)

# ── 3. Auth check ────────────────────────────────────────
print(f"\n[3] Checking authentication (GET /v1/jobs) ...")
try:
    resp = requests.get(
        f'{TINKER_API_URL}/v1/jobs',
        headers={'Authorization': f'Bearer {TINKER_API_KEY}'},
        timeout=10,
    )
    print(f"    HTTP {resp.status_code}")
    if resp.status_code == 200:
        print("    OK  — authenticated successfully")
        jobs = resp.json()
        count = len(jobs) if isinstance(jobs, list) else jobs.get('total', '?')
        print(f"    Jobs on account: {count}")
    elif resp.status_code == 401:
        print("    FAIL — 401 Unauthorized (API key is invalid or expired)")
        sys.exit(1)
    elif resp.status_code == 403:
        print("    FAIL — 403 Forbidden (key valid but no permission)")
        sys.exit(1)
    else:
        print(f"    WARN — {resp.status_code}: {resp.text[:200]}")
except Exception as e:
    print(f"    FAIL — {e}")
    sys.exit(1)

print("\n" + "=" * 55)
print("All checks passed — Tinker is ready.")
print("=" * 55)
