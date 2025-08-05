import re
from collections import defaultdict, Counter

class FractalTokenizer:
    def __init__(self, min_freq=2, max_depth=3):
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.subword_cache = {}  # Memoization for efficiency
        self.min_freq = min_freq
        self.max_depth = max_depth

    def build_vocab(self, corpus):
        words = re.findall(r'\b\w+\b|[^\w\s]', corpus.lower())  # Words + punctuation
        word_freq = Counter(words)

        vocab = [word for word, freq in word_freq.items() if freq >= self.min_freq]
        
        for i, word in enumerate(vocab, start=4):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word

    def fractal_decompose(self, word, depth=0):
        """Recursively break down words into smaller parts if they are unknown."""
        if word in self.word_to_idx or depth >= self.max_depth:
            return [self.word_to_idx.get(word, 1)]
        
        if word in self.subword_cache:
            return self.subword_cache[word]

        # Split by common patterns (vowels, consonants, or repeating characters)
        parts = re.findall(r'[aeiou]+|[^aeiou]+', word)  

        # Recursively encode parts
        encoded_parts = []
        for part in parts:
            encoded_parts.extend(self.fractal_decompose(part, depth + 1))

        self.subword_cache[word] = encoded_parts  # Cache results
        return encoded_parts

    def encode(self, text):
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())  
        encoded = []
        for word in words:
            encoded.extend(self.fractal_decompose(word))
        encoded.append(3)  # Append <EOS>
        return encoded

    def decode(self, tokens):
        return " ".join(self.idx_to_word.get(token, "<UNK>") for token in tokens if token != 0)

# Example Usage
tokenizer = FractalTokenizer(min_freq=1, max_depth=2)
corpus = "hello fractal recursion transformation"
tokenizer.build_vocab(corpus)

print("Vocab:", tokenizer.word_to_idx)
print("Encoded:", tokenizer.encode("hello fractal"))
print("Decoded:", tokenizer.decode(tokenizer.encode("hello fractal")))


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
