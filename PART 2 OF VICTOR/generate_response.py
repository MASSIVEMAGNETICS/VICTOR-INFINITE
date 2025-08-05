def generate_response(input_text, max_length=20):
    input_tensor = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
    generated = []

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            token = torch.argmax(output[:, -1, :])  # Get most probable next token
            generated.append(token.item())

            if token.item() == tokenizer.word_to_idx["<EOS>"]:
                break
    
    return tokenizer.decode(generated)

# Chatbot Testing
print("Bot:", generate_response("hello"))
print("Bot:", generate_response("tell me a joke"))
print("Bot:", generate_response("what is the meaning of life"))
print("Bot:", generate_response("do you dream"))
print("Bot:", generate_response("why is the sky blue"))
print("Bot:", generate_response("who created you"))
print("Bot:", generate_response("what is AI"))
print("Bot:", generate_response("will robots take over the world"))
print("Bot:", generate_response("what is quantum computing"))


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
