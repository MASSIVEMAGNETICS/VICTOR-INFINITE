import json

INPUT = "conversations.json"
OUTPUT = "bando_corpus.jsonl"

def parse_mapping(mapping):
    messages = []
    for node in mapping.values():
        msg = node.get("message")
        if not msg: continue
        role = msg.get("author", {}).get("role")
        content_obj = msg.get("content", {})
        parts = content_obj.get("parts", [""])
        # --- BEGIN PATCH ---
        text = ""
        if isinstance(parts, list) and parts:
            # Only keep string elements, join if necessary
            text = " ".join([str(p) for p in parts if isinstance(p, str)]).strip()
        elif isinstance(parts, str):
            text = parts.strip()
        elif isinstance(parts, dict):
            # Sometimes some wild nested stuff, try to extract
            text = str(parts)
        # --- END PATCH ---
        if text:
            messages.append({
                "id": msg.get("id"),
                "role": role,
                "text": text,
                "create_time": msg.get("create_time")
            })
    messages = sorted(messages, key=lambda x: (x["create_time"] is None, x["create_time"]))
    return messages

def export_jsonl():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for convo in data:
        mapping = convo.get("mapping", {})
        messages = parse_mapping(mapping)
        last_user = None
        # Go through all messages, pairing each user with the next assistant reply
        for m in messages:
            if m["role"] == "user":
                last_user = m["text"]
            elif m["role"] == "assistant" and last_user:
                # Write as QA pair: {"user": "...", "assistant": "..."}
                pairs.append({"user": last_user, "assistant": m["text"]})
                last_user = None  # Only pair once
    print(f"Exported {len(pairs)} user→assistant pairs to {OUTPUT}")
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for pair in pairs:
            json.dump(pair, out, ensure_ascii=False)
            out.write("\n")

if __name__ == "__main__":
    export_jsonl()
