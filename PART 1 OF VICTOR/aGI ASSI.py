# ======= FLASK WEB UI (DROP-IN AGI ASSISTANT) =======
try:
    from flask import Flask, request, render_template_string
except ImportError:
    print("Install Flask: pip install flask")
    raise

HTML = """
<!doctype html>
<html>
<head>
<title>Victor AGI Web Assistant</title>
<style>
body { background: #161616; color: #fff; font-family: monospace; margin: 2em;}
textarea,input { background: #222; color: #fff; border: 1px solid #444; }
button { background: #272; color: #fff; border: none; padding: 8px 20px;}
pre { background: #191919; padding: 10px;}
</style>
</head>
<body>
<h2>🧠 Victor AGI Web Assistant</h2>
<form method="post">
    <textarea name="query" rows="3" cols="80" placeholder="Ask, teach, upload, summarize, web search..."></textarea><br>
    <button type="submit">Send</button>
</form>
{% if result %}
<pre><b>Victor:</b> {{ result }}</pre>
{% endif %}
<h4>Memory (last 10):</h4>
<pre>
{% for item in mem[-10:] %}
{{ loop.index }}: {{ item[:100] }}
{% endfor %}
</pre>
</body>
</html>
"""

def run_flask_agi():
    global agent  # Use your VictorAgiResearchAgent instance
    app = Flask(__name__)
    @app.route("/", methods=["GET", "POST"])
    def home():
        result = ""
        if request.method == "POST":
            user_in = request.form["query"].strip()
            # Reuse all CLI commands here (see your superloop logic)
            if user_in.lower() == "summary":
                context = agent.context_summary()
                result = f"Context summary vector: {context[:8]} ..."
            elif user_in.lower() == "mem":
                result = f"Victor has memorized {len(agent.texts)} things."
            elif user_in.lower().startswith("recall "):
                q = user_in.split(" ", 1)[1] if " " in user_in else ""
                results = agent.comprehend(q, topk=3)
                result = "<br>".join([f"({score:.3f}) {txt}" for txt, score in results])
            elif user_in.lower().startswith("web "):
                q = user_in[4:].strip()
                try:
                    snippet = search_duckduckgo(q)
                    result = f"Web: {snippet}"
                    agent.add_knowledge(f"Web: {q} => {snippet}")
                except Exception as e:
                    result = f"Web search failed: {e}"
            elif user_in.lower().startswith("summarize "):
                text = user_in[10:].strip()
                out = agent.nlp.summary(agent.nlp.embed(text))
                result = f"Summary vector (first 8): {out.data.flatten()[:8]}"
                agent.add_knowledge(f"Summary: {text[:80]}")
            elif user_in.lower().startswith("memorizefile "):
                path = user_in.split(" ", 1)[1].strip()
                if not os.path.exists(path):
                    result = "Victor: File not found."
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    for l in lines:
                        agent.add_knowledge(l.strip())
                    result = f"Victor: Memorized {len(lines)} lines from {path}."
            elif user_in.lower().startswith("export "):
                fname = user_in.split(" ",1)[1].strip()
                with open(fname, "w", encoding="utf-8") as f:
                    for t in agent.texts:
                        f.write(t+"\n")
                result = f"Victor: Memory exported to {fname}"
            elif user_in.lower().startswith("import "):
                fname = user_in.split(" ",1)[1].strip()
                if os.path.exists(fname):
                    with open(fname, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    for l in lines:
                        agent.add_knowledge(l.strip())
                    result = f"Victor: Imported {len(lines)} knowledge items."
                else:
                    result = "Victor: File not found."
            elif user_in.lower() == "reflect":
                summary = agent.context_summary()
                result = f"Reflection vector: {summary[:8]}"
            elif user_in.lower() == "autoevolve":
                out_list = []
                for txt in agent.texts[-20:]:
                    summary = agent.nlp.summary(agent.nlp.embed(txt))
                    out_list.append(f"{txt[:40]}...: {summary.data.flatten()[:8]}")
                result = "<br>".join(out_list)
            elif user_in.lower() == "graph":
                with open("victor_graph.dot", "w") as f:
                    f.write("digraph VictorKnowledge {\n")
                    for i, t in enumerate(agent.texts):
                        if i>0:
                            f.write(f'  "{agent.texts[i-1][:20]}" -> "{t[:20]}";\n')
                    f.write("}\n")
                result = "Victor: Knowledge graph exported to victor_graph.dot"
            else:
                agent.add_knowledge(user_in)
                results = agent.comprehend(user_in, topk=1)
                if results:
                    result = f"Closest recall: '{results[0][0]}' (Score: {results[0][1]:.3f})"
                else:
                    result = "I don't know yet, but I'm learning. Feed me more."
        return render_template_string(HTML, result=result, mem=agent.texts)
    app.run(host="0.0.0.0", port=9888, debug=False)

# Add to your CLI menu, at the very bottom of your file, this way:
if __name__ == "__main__":
    # ... CLI superloop code ...
    while True:
        user_in = input("👤 You: ").strip()
        if user_in.lower() == "webui":
            print("Launching Victor AGI Web UI on http://localhost:9888 ...")
            run_flask_agi()
            break
        # ... rest of your loop ...

