# training_loop.py
# VICTOR OMNIFRACTAL GENESIS 5.0 TRAINING ENGINE
# Architect: Brandon & Tori

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from core_model import OmniFractalCore
from error_sentinel import safe_execute
import os
import json


class Config:
    """
    Dynamic Config Loader from victor_training_package.json
    """

    @staticmethod
    def load(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        cfg = type('VictorConfig', (), {})
        for key, value in data.items():
            setattr(cfg, key, value)
        return cfg


class VictorTrainer:
    """
    Adaptive Evolutionary Trainer for Victor AI
    Integrates Recursive Growth, Self-Repair, Emotional Logging
    Version: 5.0.OMNIFRACTAL
    """

    def __init__(self, config_path='data/training_package/victor_training_package.json'):
        self.cfg = Config.load(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = OmniFractalCore(vocab_size=self.cfg.vocab_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

        os.makedirs(self.cfg.log_dir, exist_ok=True)
        os.makedirs(self.cfg.memory_dir, exist_ok=True)

    def train_step(self, batch):
        text_input, target = batch
        text_input, target = text_input.to(self.device), target.to(self.device)

        try:
            output = self.model(text_input)
            loss = self.loss_fn(output.view(-1, output.size(-1)), target.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

        except Exception as e:
            safe_execute(e)
            return None

    def train(self, dataset, epochs=1, batch_size=4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                loss = self.train_step(batch)
                if loss:
                    total_loss += loss

            self.log(f"Epoch {epoch + 1} completed. Avg Loss: {total_loss / len(dataloader)}")

    def log(self, message):
        print("[Victor Log]", message)
        with open(os.path.join(self.cfg.log_dir, 'victor_training_log.txt'), 'a') as f:
            f.write(message + '\n')


# === Example Usage ===
if __name__ == "__main__":
    from dummy_dataset import DummyVictorDataset  # Replace with real dataset

    trainer = VictorTrainer()
    dataset = DummyVictorDataset(vocab_size=trainer.cfg.vocab_size, seq_len=128, size=1000)

    trainer.train(dataset, epochs=5, batch_size=8)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
