import os
import sys

# Test 1: Check if dotenv is working
print("=== TESTING API KEY SETUP ===")
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded successfully")
except Exception as e:
    print(f"❌ dotenv error: {e}")

# Test 2: Check if key is being read
key = os.getenv("ANTHROPIC_API_KEY")
print(f"\n=== API KEY CHECK ===")
if key:
    print(f"✅ Key found: {key[:15]}...")
    print(f"✅ Key length: {len(key)}")
    print(f"✅ Starts with 'sk-ant-': {key.startswith('sk-ant-')}")
else:
    print("❌ No API key found in environment")

# Test 3: Check current directory and .env file
print(f"\n=== FILE SYSTEM CHECK ===")
print(f"Current directory: {os.getcwd()}")
print(f".env file exists: {os.path.exists('.env')}")

if os.path.exists('.env'):
    print("\n=== .env FILE CONTENTS ===")
    with open('.env', 'r') as f:
        content = f.read()
    # Only show first part of key for security
    lines = content.strip().split('\n')
    for line in lines:
        if 'ANTHROPIC_API_KEY' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                key_part = parts[1].strip('"').strip("'")
                print(f"Found in .env: {key_part[:15]}...")

# Test 4: Try Claude API call
print(f"\n=== CLAUDE API TEST ===")
if key:
    try:
        import anthropic
        print("✅ anthropic library imported")
        
        client = anthropic.Anthropic(api_key=key)
        print("✅ client created")
        
        # Simple test call
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}]
        )
        print("✅ API call successful!")
        print(f"Response: {response.content[0].text}")
        
    except ImportError:
        print("❌ anthropic library not installed. Run: pip install anthropic")
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print(f"Error type: {type(e).__name__}")
else:
    print("❌ No key to test")

print(f"\n=== PYTHON ENVIRONMENT ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")