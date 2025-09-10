import os
import re
import requests
import streamlit as st
from typing import Optional, List
from bs4 import BeautifulSoup

# ---------- Optional: .env or secrets ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- URL reader (robust: headers + proxy fallback) ----------
@st.cache_data(show_spinner=False, ttl=300)
def read_url_content(url: str) -> Optional[str]:
    def _clean(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join([ln for ln in lines if ln])

    # 1) Try direct fetch (browser-like headers)
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=25)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        text = soup.get_text(separator="\n")
        cleaned = _clean(text)
        if cleaned:
            return cleaned
    except requests.RequestException:
        pass

    # 2) Fallback: simple text proxy (handles many JS/blocked pages)
    try:
        if not re.match(r"^https?://", url, flags=re.I):
            url = "https://" + url
        proxy_url = f"https://r.jina.ai/{url}"
        prox = requests.get(proxy_url, timeout=25)
        if prox.ok:
            txt = prox.text.strip()
            cleaned = _clean(txt)
            if cleaned:
                return cleaned
    except requests.RequestException:
        pass

    return None

# ---------- API keys ----------
def get_key(provider: str) -> Optional[str]:
    env_var = {
        "ChatGPT": "OPENAI_API_KEY",
        "Claude":  "ANTHROPIC_API_KEY",
        "Gemini":  "GEMINI_API_KEY",
    }[provider]

    # First try environment variable
    key = os.getenv(env_var)
    if key and key.strip():
        return key.strip()

    # Then try Streamlit secrets
    try:
        key = st.secrets.get(env_var)
        if key and key.strip():
            return key.strip()
    except Exception:
        pass
    
    return None

# ---------- Prompt building ----------
def build_instructions(style_key: str) -> str:
    if style_key == "100_words":
        return "Summarize the document in ~100 words (max 110)."
    elif style_key == "two_paragraphs":
        return "Summarize the document in exactly two connected paragraphs."
    return "Summarize the document in exactly five bullet points."

def build_prompt(document_text: str, style_key: str, language: str) -> str:
    style_instructions = build_instructions(style_key)
    return (
        f"{style_instructions}\n\n"
        f"Write the output in **{language}**.\n"
        "Source text:\n\n---\n"
        f"{document_text}"
    )

# ---------- Provider calls ----------
def summarize_openai(model: str, prompt: str, key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def validate_openai(key: str) -> None:
    if not key or not key.strip():
        raise ValueError("OpenAI API key is empty or None")
    
    from openai import OpenAI
    client = OpenAI(api_key=key)
    _ = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "ping"}],
        temperature=0,
    )

def summarize_claude(model: str, prompt: str, key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return "".join(getattr(block, "text", "") for block in msg.content).strip()

def validate_claude(key: str) -> None:
    if not key or not key.strip():
        raise ValueError("Claude API key is empty or None")
    
    # Check if key has proper format (should start with 'sk-ant-')
    if not key.startswith('sk-ant-'):
        raise ValueError("Claude API key should start with 'sk-ant-'. Please check your API key format.")
    
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Claude SDK not installed. Run: pip install anthropic")
    
    client = anthropic.Anthropic(api_key=key)
    try:
        _ = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
        )
    except Exception as e:
        if "authentication_error" in str(e).lower():
            raise ValueError("Invalid Claude API key. Please check that your API key is correct and active.")
        raise

def summarize_gemini(model: str, prompt: str, key: str) -> str:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    genai.configure(api_key=key)

    def _try(model_name: str) -> str:
        m = genai.GenerativeModel(model_name)
        resp = m.generate_content(prompt)
        return (resp.text or "").strip()

    try:
        return _try(model)
    except ResourceExhausted:
        if model != "gemini-1.5-flash":
            return _try("gemini-1.5-flash")
        raise
    except Exception as e:
        tb = str(e).lower()
        if "429" in tb or "quota" in tb or "resourceexhausted" in tb:
            return _try("gemini-1.5-flash")
        raise

def validate_gemini(key: str) -> None:
    if not key or not key.strip():
        raise ValueError("Gemini API key is empty or None")
    
    import google.generativeai as genai
    genai.configure(api_key=key)
    _ = genai.GenerativeModel("gemini-1.5-flash").generate_content("ping")

def call_provider(provider: str, model: str, prompt: str, key: str) -> str:
    if provider == "ChatGPT":
        return summarize_openai(model, prompt, key)
    if provider == "Claude":
        return summarize_claude(model, prompt, key)
    if provider == "Gemini":
        return summarize_gemini(model, prompt, key)
    raise ValueError("Unsupported provider")

def validate_key(provider: str, key: str):
    if provider == "ChatGPT":
        return validate_openai(key)
    if provider == "Claude":
        return validate_claude(key)
    if provider == "Gemini":
        return validate_gemini(key)

# ---------- Session cache: validate once per provider ----------
if "validated_providers" not in st.session_state:
    st.session_state.validated_providers = set()

def validate_key_once(provider: str, key: str):
    # Always validate if key has changed
    cache_key = f"{provider}_{key[:10] if key else 'none'}"
    if cache_key in st.session_state.get('validated_cache', set()):
        return
    
    validate_key(provider, key)
    
    if 'validated_cache' not in st.session_state:
        st.session_state.validated_cache = set()
    st.session_state.validated_cache.add(cache_key)
    st.session_state.validated_providers.add(provider)

# ---------- Chunking (map -> reduce) ----------
PROVIDER_CHUNK_SIZE = {
    "ChatGPT": 9000,
    "Claude":  7000,
    "Gemini":  6000,
}
OVERLAP = 400

def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    chunks = []
    n = len(text)
    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end == n:
            break
        i = end - overlap if overlap > 0 else end
    return chunks

def summarize_long_text(provider: str, model: str, full_text: str, style_key: str, language: str, key: str) -> str:
    chunk_size = PROVIDER_CHUNK_SIZE.get(provider, 9000)
    parts = chunk_text(full_text, chunk_size=chunk_size, overlap=OVERLAP)

    partial_summaries = []
    for idx, part in enumerate(parts, 1):
        prompt_part = (
            "Summarize the following PART of a larger document into 4-6 compact bullet points.\n\n---\n" + part
        )
        partial = call_provider(provider, model, prompt_part, key)
        partial_summaries.append(f"Part {idx} summary:\n{partial}")

    combined_source = "\n\n".join(partial_summaries)
    final_prompt = build_prompt(combined_source, style_key, language)
    final = call_provider(provider, model, final_prompt, key)
    return final

def generate_summary(provider: str, model: str, text: str, style_key: str, language: str, key: str) -> str:
    if len(text) <= PROVIDER_CHUNK_SIZE.get(provider, 9000):
        prompt = build_prompt(text, style_key, language)
        return call_provider(provider, model, prompt, key)
    else:
        return summarize_long_text(provider, model, text, style_key, language, key)

# ---------- Model catalog (Less expensive vs Latest) ----------
MODEL_CATALOG = {
    "ChatGPT": {
        "less_expensive": "gpt-4o-mini",      # budget
        "latest":         "gpt-4o",           # premium/latest stable
    },
    "Claude": {
        "less_expensive": "claude-3-haiku-20240307",
        "latest":         "claude-3-5-sonnet-20240620",
    },
    "Gemini": {
        "less_expensive": "gemini-1.5-flash",
        "latest":         "gemini-1.5-pro",
    },
}

# ---------- UI ----------
st.title("🌐 HW 2 — URL Summarizer (ChatGPT · Claude · Gemini)")

url = st.text_input("Enter a URL to summarize:", placeholder="https://example.com/article")
language = st.selectbox("Output language:", ["English", "Hebrew", "Spanish"], index=0)
st.caption("Paste a full URL. The app fetches the page text and summarizes it in your chosen language.")

with st.sidebar:
    st.header("Summary Options")
    summary_choice = st.radio(
        "Choose a summary type:",
        [
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points",
        ],
        index=0,
    )

    st.divider()
    st.header("Provider & Model")

    provider = st.selectbox("LLM Provider", ["ChatGPT", "Claude", "Gemini"], index=0)

    tier = st.radio(
        "Model tier",
        ["Less expensive", "Latest"],
        help="Switch between low-cost and premium/latest models for the selected provider.",
        index=0,
    )

    # Bind tier -> concrete model name
    model = MODEL_CATALOG[provider]["less_expensive" if tier == "Less expensive" else "latest"]
    st.write(f"**Using model:** `{model}`")

style_key = (
    "100_words"
    if summary_choice.startswith("Summarize the document in 100 words")
    else "two_paragraphs"
    if "2 connecting paragraphs" in summary_choice
    else "five_bullets"
)

# ---------- Reactive run ----------
if url.strip():
    with st.spinner("Fetching URL…"):
        text = read_url_content(url.strip())

    if not text:
        st.error("Could not extract any text from that URL.")
    else:
        chosen_provider = provider
        key = get_key(chosen_provider)
        
        if not key:
            st.error(
                f"Missing API key for **{chosen_provider}**. "
                "Set it in `.env` or Streamlit secrets. "
                f"Expected variable: `{ {'ChatGPT':'OPENAI_API_KEY','Claude':'ANTHROPIC_API_KEY','Gemini':'GEMINI_API_KEY'}[chosen_provider] }`"
            )
            
            # Show helpful instructions
            if chosen_provider == "Claude":
                st.info("""
                **To get a Claude API key:**
                1. Go to https://console.anthropic.com/
                2. Sign up or log in to your account
                3. Navigate to API Keys section
                4. Create a new API key
                5. The key should start with 'sk-ant-'
                """)
        else:
            try:
                with st.spinner(f"Validating {chosen_provider} key…"):
                    validate_key_once(chosen_provider, key)
                st.success(f"{chosen_provider} key validated ✅")
            except ValueError as e:
                st.error(f"**{chosen_provider} API Key Error:** {str(e)}")
                if chosen_provider == "Claude":
                    st.info("""
                    **Common Claude API key issues:**
                    - Make sure your key starts with 'sk-ant-'
                    - Check that you copied the entire key without extra spaces
                    - Verify the key is active in your Anthropic Console
                    - Make sure you have credits/usage remaining
                    """)
                st.stop()
            except Exception as e:
                st.error(f"{chosen_provider} validation failed: {str(e)}")
                st.exception(e)
                st.stop()

            # Summarize
            st.subheader("Summary")
            try:
                with st.spinner(f"Summarizing with {chosen_provider} · {model}…"):
                    output = generate_summary(chosen_provider, model, text, style_key, language, key)
                st.write(output)

                # Gentle note about Gemini fallback if user chose Pro
                if chosen_provider == "Gemini" and model == "gemini-1.5-pro":
                    st.caption("Note: If Gemini hit free-tier quotas, the app may automatically fall back to **gemini-1.5-flash**.")

                st.caption(
                    f"Provider: `{chosen_provider}` · Tier: **{tier}** · Model: `{model}` · "
                    f"Style: **{summary_choice}** · Language: **{language}**"
                )
            except Exception as e:
                st.error(f"{chosen_provider} error while generating the summary.")
                st.exception(e)