from dotenv import load_dotenv
import os

load_dotenv()

key = os.environ.get("ANTHROPIC_API_KEY")

if key is None:
    print("ANTHROPIC_API_KEY was not found at all — check the variable name in .env")
else:
    print(f"Key length: {len(key)}")
    print(f"Starts with: {key[:15]}")
    print(f"Ends with: {key[-5:]}")
    print(f"Has leading/trailing whitespace: {key != key.strip()}")
    print(f"Contains quote characters: {'\"' in key or chr(39) in key}")
