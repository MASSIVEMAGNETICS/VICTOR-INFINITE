import json

with open("conversations.json", "r", encoding="utf-8") as f:
    # Read just the first 1000 characters (should get the start of the file)
    start = f.read(1000)
    print("First 1000 characters:\n")
    print(start)
    f.seek(0)
    # Try to load just the start of the JSON
    try:
        obj = json.load(f)
        print("\nType at root:", type(obj))
        if isinstance(obj, list):
            print("First element type:", type(obj[0]))
            print("First element:", obj[0])
        elif isinstance(obj, dict):
            print("Keys at root:", list(obj.keys()))
            # If it has a 'messages' or 'conversations' key, print first entry
            for key in ['messages','conversations','data','logs']:
                if key in obj:
                    print(f"First item in {key}:", obj[key][0])
    except Exception as e:
        print(f"Could not fully load JSON: {e}")
