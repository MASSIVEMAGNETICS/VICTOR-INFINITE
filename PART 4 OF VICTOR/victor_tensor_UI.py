--- /dev/null
+++ b/c:\Users\Brand Name\Desktop\victorCH\victor_tensor_ui.py
@@ -0,0 +1,147 @@
+#!/usr/bin/env python3
+# -*- coding: utf-8 -*-
+
+"""
+FILE: victor_tensor_ui.py
+PURPOSE: Streamlit UI for victor_tensor_v4.py
+"""
+
+import streamlit as st
+import numpy as np
+import pandas as pd
+
+# Assuming victor_tensor_v4.py is in the same directory
+from victor_tensor_v4 import (
+    Tensor, Linear, ReLU, Sigmoid, Tanh, Sequential,
+    MSELoss, MAELoss, SGD, Trainer
+)
+
+def get_activation_module(name_str):
+    """Returns an activation module instance based on its name."""
+    if name_str == "ReLU":
+        return ReLU()
+    elif name_str == "Sigmoid":
+        return Sigmoid()
+    elif name_str == "Tanh":
+        return Tanh()
+    elif name_str == "None":
+        return None
+    raise ValueError(f"Unknown activation function: {name_str}")
+
+def get_loss_function(name_str):
+    """Returns a loss function instance based on its name."""
+    if name_str == "MSELoss":
+        return MSELoss()
+    elif name_str == "MAELoss":
+        return MAELoss()
+    raise ValueError(f"Unknown loss function: {name_str}")
+
+st.set_page_config(layout="wide")
+st.title("🔱 VictorTensor Neural Network Trainer 🔱")
+
+# --- Sidebar for Configuration ---
+st.sidebar.title("Configuration")
+
+st.sidebar.header("Data Parameters")
+num_samples = st.sidebar.slider("Number of Samples", 50, 1000, 100, key="num_samples")
+input_features = st.sidebar.number_input("Input Features", 1, 20, 2, key="input_features")
+# Output features fixed to 1 for this binary classification example
+output_features = 1
+
+st.sidebar.header("Model Architecture")
+st.sidebar.markdown(f"Input Layer: {input_features} features")
+hidden_units = st.sidebar.number_input("Hidden Layer Units", 4, 256, 32, key="hidden_units")
+hidden_activation_name = st.sidebar.selectbox("Hidden Activation", ["ReLU", "Sigmoid", "Tanh"], key="hidden_act")
+st.sidebar.markdown(f"Output Layer: {output_features} feature(s)")
+output_activation_name = st.sidebar.selectbox("Output Activation", ["Sigmoid", "None"], index=0, key="output_act")
+
+st.sidebar.header("Training Parameters")
+epochs = st.sidebar.number_input("Epochs", 1, 5000, 100, key="epochs")
+learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.01, format="%.4f", key="lr")
+momentum = st.sidebar.slider("Momentum (SGD)", 0.0, 0.99, 0.9, format="%.2f", key="momentum")
+loss_fn_name = st.sidebar.selectbox("Loss Function", ["MSELoss", "MAELoss"], key="loss_fn")
+
+train_button = st.sidebar.button("🚀 Train Model 🚀")
+
+# --- Main Area for Output ---
+status_placeholder = st.empty()
+col1, col2 = st.columns(2)
+loss_chart_placeholder = col1.empty()
+results_placeholder = col2.empty()
+
+if 'loss_history' not in st.session_state:
+    st.session_state.loss_history = []
+
+if train_button:
+    st.session_state.loss_history = [] # Reset loss history for new training
+
+    # 1. Generate Data (simple binary classification task)
+    X_data = np.random.randn(num_samples, input_features).astype(np.float32)
+    Y_data = np.random.randint(0, 2, (num_samples, output_features)).astype(np.float32)
+    X_tensor = Tensor(X_data)
+    Y_tensor = Tensor(Y_data)
+
+    # 2. Create Model
+    layers = []
+    layers.append(Linear(input_features, hidden_units))
+    hidden_activation_module = get_activation_module(hidden_activation_name)
+    if hidden_activation_module:
+        layers.append(hidden_activation_module)
+    
+    layers.append(Linear(hidden_units, output_features))
+    output_activation_module = get_activation_module(output_activation_name)
+    if output_activation_module:
+        layers.append(output_activation_module)
+    
+    model = Sequential(*layers)
+
+    # 3. Optimizer and Loss
+    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
+    loss_fn = get_loss_function(loss_fn_name)
+    trainer = Trainer(model, optimizer, loss_fn)
+
+    # 4. Training Loop
+    loss_plot = loss_chart_placeholder.line_chart(pd.DataFrame(columns=['epoch', 'loss']).set_index('epoch'))
+
+    for epoch in range(epochs):
+        loss = trainer.train_step(X_tensor, Y_tensor)
+        current_loss_value = loss.data.item() # Get scalar from 0-dim np array
+        st.session_state.loss_history.append(current_loss_value)
+        
+        status_placeholder.text(f"Epoch {epoch+1}/{epochs} | Loss: {current_loss_value:.6f}")
+        
+        # Update plot: create a new DataFrame for each update
+        loss_df = pd.DataFrame({
+            'epoch': range(1, len(st.session_state.loss_history) + 1),
+            'loss': st.session_state.loss_history
+        }).set_index('epoch')
+        loss_plot.line_chart(loss_df)
+
+    status_placeholder.success(f"🎉 Training Complete! Final Loss: {st.session_state.loss_history[-1]:.6f}")
+
+    # 5. Display Results
+    with results_placeholder.container():
+        st.subheader("Model Predictions (Sample)")
+        predictions_tensor = model(X_tensor)
+        predictions_data = predictions_tensor.data
+        
+        # For binary classification with Sigmoid, round predictions
+        if output_activation_name == "Sigmoid":
+            predicted_classes = (predictions_data > 0.5).astype(int)
+        else:
+            # For regression or no specific output activation, show raw output
+            predicted_classes = predictions_data 
+
+        # Display a few samples
+        num_display_samples = min(10, num_samples)
+        display_df = pd.DataFrame()
+        for i in range(min(input_features, 5)): # Show up to 5 input features
+            display_df[f'Input_{i+1}'] = X_data[:num_display_samples, i]
+        display_df['Target'] = Y_data[:num_display_samples, 0] # Assuming single output
+        display_df['Prediction_Raw'] = predictions_data[:num_display_samples, 0]
+        if output_activation_name == "Sigmoid":
+             display_df['Prediction_Class'] = predicted_classes[:num_display_samples, 0]
+
+        st.dataframe(display_df)
+else:
+    if not st.session_state.loss_history: # Show placeholder text if no training has happened yet
+        loss_chart_placeholder.markdown("📈 *Loss chart will appear here after training.*")
+        results_placeholder.markdown("📊 *Sample predictions will appear here after training.*")

