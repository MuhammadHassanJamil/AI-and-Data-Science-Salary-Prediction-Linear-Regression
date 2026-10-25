"""
Quick entry point for SalaryModelTrainer
Usage:
    python train.py
"""
from trainer import SalaryModelTrainer

if __name__ == "__main__":
    trainer = SalaryModelTrainer()
    trainer.run_pipeline()
