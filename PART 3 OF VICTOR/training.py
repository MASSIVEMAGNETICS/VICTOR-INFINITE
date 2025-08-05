import torch
import torch.nn as nn
import torch.optim as optim
from fractal_transformer import FractalTransformer

# Define model parameters
VOCAB_SIZE = len(tokenizer.word_to_idx)
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4  # Increased layers for better learning
FF_HIDDEN_DIM = 512
RECURSION_DEPTH = 3  # Slightly deeper recursion
MAX_SEQ_LEN = 20  # Longer sequence length

# Initialize the model
model = FractalTransformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, FF_HIDDEN_DIM, RECURSION_DEPTH, MAX_SEQ_LEN)
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower LR for stability
criterion = nn.CrossEntropyLoss()

# Train the model on the new dataset
EPOCHS = 2000  # More training epochs for better learning
for epoch in range(EPOCHS):
    total_loss = 0
    for input_tensor, target_tensor in zip(input_tensors, target_tensors):
        optimizer.zero_grad()
        
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        target_tensor = target_tensor.unsqueeze(0)
        
        output = model(input_tensor)
        loss = criterion(output.squeeze(0), target_tensor)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

print("Training complete!")


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
