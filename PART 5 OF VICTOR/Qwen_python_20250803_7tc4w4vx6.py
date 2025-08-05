from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Qwen
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1.5-0.5B")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1.5-0.5B").eval()

# Load Victor (assuming you've defined it from the file)
victor = VictorASIFractalLightModelAdvanced(vocab_size=65536, dim=1024).eval()

# Example input
prompt = "Explain the meaning of life in three perspectives."
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Step 1: Generate hidden states from Qwen
with torch.no_grad():
    outputs = qwen(
        input_ids=inputs.input_ids,
        output_hidden_states=True,
        return_dict=True
    )
    last_hidden_state = outputs.hidden_states[-1]  # (B, L, D_qwen)

# Step 2: Project Qwen's hidden states into Victor’s space
# You need a projection layer because dimensions likely differ
D_qwen = last_hidden_state.shape[-1]  # e.g., 896
D_victor = 1024

# Trainable projection (can be frozen or fine-tuned)
proj_layer = nn.Linear(D_qwen, D_victor)
projected_states = proj_layer(last_hidden_state)  # Now (B, L, 1024)

# Step 3: Run Victor’s consensus engine
# Note: Victor expects token IDs, so we fake it by passing dummy IDs
# But we override the embedding with projected_states
with torch.no_grad():
    # Hack: Replace Victor's embed forward temporarily
    original_embed = victor.embed
    victor.embed = lambda x: projected_states  # Bypass fractal embed
    victor_out = victor(token_ids=torch.zeros_like(inputs.input_ids))  # Dummy input
    refined_states = victor_out["gen_logits"]  # Or use tool_logits, or h from memory
    victor.embed = original_embed  # Restore