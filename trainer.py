"""
Model Trainer & Dataset Splitter for AI & Data Science Salary Prediction
========================================================================
Machine Learning Fundamentals (AIC354) - Assignment 3 (Task 2)
Supervised by: Dr. Rao Muhammad Adeel Nawab
COMSATS University Islamabad, Lahore Campus

This script handles:
1. Splitting the raw dataset into 2/3 Training and 1/3 Testing partitions
   and saving them to 'train_data.csv' and 'test_data.csv'.
2. The Model Trainer class ('SalaryModelTrainer') that trains, evaluates,
   and compares 5 distinct Linear Regression algorithms:
     - Lasso Regression (L1)
     - Ridge Regression (L2)
     - Huber Regressor (Robust to Outliers)
     - Bayesian Ridge Regression (Probabilistic)
     - ElasticNet Regression (L1 + L2)
3. 10-Fold Cross Validation (K=10) on the full dataset (~8,000 records),
   matching the assignment requirement (Task 2, step 3).
4. Exporting per-fold evaluation table to 'kfold_10_evaluation_table.csv'.
5. Exporting summary metrics to 'model_accuracy_summary.csv'.
6. Exporting the best trained pipeline to 'salary_prediction_model.pkl'.
7. Generating test predictions and saving them to 'test_predictions.csv'.
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
)

class SalaryModelTrainer:
    """
    End-to-end Trainer for evaluating and deploying regularized linear models
    using 10-Fold Cross-Validation and 2/3:1/3 Holdout Partitioning.
    """
    def __init__(self, data_path=None, output_dir=None, n_splits=10, random_state=42):
        self.random_state = random_state
        self.n_splits = n_splits
        self.output_dir = Path(output_dir or Path(__file__).resolve().parent)
        self.data_path = Path(data_path or self.output_dir / "ai_ds_salaries_DUMMY.csv")
        
        # Feature definitions
        self.numeric_features = ["work_year", "remote_ratio"]
        self.categorical_features = [
            "experience_level",
            "employment_type",
            "job_title",
            "employee_residence",
            "company_location",
            "company_size",
        ]
        self.target_column = "salary_in_usd"

        # Preprocessor definition
        self.preprocessor = self._build_preprocessor()

        # 5 Linear Regression algorithms (none used in Car Price Prediction)
        self.algorithms = {
            "Lasso Regression": Lasso(alpha=10.0, max_iter=2000, random_state=self.random_state),
            "Ridge Regression": Ridge(alpha=1.0, random_state=self.random_state),
            "Huber Regressor": HuberRegressor(max_iter=1000),
            "Bayesian Ridge": BayesianRidge(),
            "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
        }

        self.df_full = None
        self.train_df = None
        self.test_df = None
        self.best_model_name = None
        self.best_pipeline = None
        self.per_fold_table = None
        self.summary_table = None

    def _build_preprocessor(self):
        """Constructs leakage-free ColumnTransformer pipeline."""
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer([
            ("num", num_pipe, self.numeric_features),
            ("cat", cat_pipe, self.categorical_features),
        ])

    def load_and_split_data(self, test_size=1/3):
        """
        Loads dataset and splits into 2/3 Training (66.7%) and 1/3 Testing (33.3%).
        Saves split datasets to 'train_data.csv' and 'test_data.csv'.
        """
        print(f"\n[1/6] Loading dataset from: {self.data_path}")
        self.df_full = pd.read_csv(self.data_path)
        total_rows = len(self.df_full)
        print(f"      Total records: {total_rows:,} rows, {self.df_full.shape[1]} columns")

        # Perform 2/3 Train, 1/3 Test Split
        train_df, test_df = train_test_split(
            self.df_full,
            test_size=test_size,
            random_state=self.random_state,
            shuffle=True
        )

        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        train_path = self.output_dir / "train_data.csv"
        test_path = self.output_dir / "test_data.csv"

        self.train_df.to_csv(train_path, index=False)
        self.test_df.to_csv(test_path, index=False)

        print(f"\n[2/6] Performed 2/3 Train & 1/3 Test Partition:")
        print(f"      - Training Set (2/3): {len(self.train_df):,} rows ({len(self.train_df)/total_rows*100:.1f}%) -> {train_path.name}")
        print(f"      - Testing Set  (1/3): {len(self.test_df):,} rows ({len(self.test_df)/total_rows*100:.1f}%) -> {test_path.name}")

        return self.train_df, self.test_df

    def run_10_fold_cross_validation(self):
        """
        Executes 10-Fold Cross-Validation on the full 8,000 dataset for all 5 algorithms,
        matching Task 2 step 3 of Assignment 3.
        """
        if self.df_full is None:
            self.df_full = pd.read_csv(self.data_path)

        X = self.df_full[self.numeric_features + self.categorical_features]
        y = self.df_full[self.target_column]

        K = self.n_splits  # 10 folds
        kfold = KFold(n_splits=K, shuffle=True, random_state=self.random_state)
        folds = list(kfold.split(X))

        print(f"\n[3/6] Running {K}-Fold Cross-Validation on Full Dataset ({len(X):,} records):")
        print(f"      Each fold evaluates on ~{len(folds[0][1]):,} test records and trains on ~{len(folds[0][0]):,} records.")
        print("-" * 90)

        fold_rmse = {}
        timing = {}

        for name, algo in self.algorithms.items():
            base_model = Pipeline([
                ("preprocessor", self._build_preprocessor()),
                ("regressor", algo)
            ])

            scores = []
            t0 = time.time()
            for fold_num, (tr_idx, te_idx) in enumerate(folds, start=1):
                fold_model = clone(base_model)
                fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                pred = fold_model.predict(X.iloc[te_idx])
                rmse = np.sqrt(mean_squared_error(y.iloc[te_idx], pred))
                scores.append(rmse)

            timing[name] = time.time() - t0
            fold_rmse[name] = np.array(scores)
            mean_s = np.mean(scores)
            std_s = np.std(scores)
            min_s = np.min(scores)
            max_s = np.max(scores)

            print(f"  {name:<24} -> Mean RMSE: ${mean_s:>9,.2f} (±${std_s:>6,.2f}) | Range: [${min_s:>9,.2f} - ${max_s:>9,.2f}] ({timing[name]:.2f}s)")

        print("-" * 90)

        # Build Per-Fold Table
        per_fold_df = pd.DataFrame(fold_rmse)
        per_fold_df.index = [f"Fold {i}" for i in range(1, K + 1)]
        per_fold_df.index.name = "Fold"
        self.per_fold_table = per_fold_df

        per_fold_path = self.output_dir / f"kfold_{K}_evaluation_table.csv"
        per_fold_df.to_csv(per_fold_path)
        print(f"      Saved per-fold results to: {per_fold_path.name}")

        return self.per_fold_table, timing

    def train_and_evaluate(self):
        """
        Combines 10-Fold CV metrics with 2/3:1/3 Holdout test metrics.
        """
        if self.train_df is None or self.test_df is None:
            self.load_and_split_data()

        # Run 10-fold CV
        per_fold_df, cv_timing = self.run_10_fold_cross_validation()

        X_train = self.train_df[self.numeric_features + self.categorical_features]
        y_train = self.train_df[self.target_column]
        X_test = self.test_df[self.numeric_features + self.categorical_features]
        y_test = self.test_df[self.target_column]

        print(f"\n[4/6] Evaluating on 1/3 Holdout Test Set ({len(X_test):,} instances):")
        print("-" * 95)
        print(f"{'Algorithm':<24} | {'10-Fold RMSE':<14} | {'Holdout R²':<11} | {'Holdout RMSE':<14} | {'Holdout MAE':<13}")
        print("-" * 95)

        results = []
        for name, algo in self.algorithms.items():
            pipe = Pipeline([
                ("preprocessor", self._build_preprocessor()),
                ("regressor", algo)
            ])
            t0 = time.time()
            pipe.fit(X_train, y_train)
            h_time = time.time() - t0

            y_pred = pipe.predict(X_test)
            h_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            h_mae = mean_absolute_error(y_test, y_pred)
            h_r2 = r2_score(y_test, y_pred)
            
            # Percentage error metrics
            pct_err = np.abs(y_test - y_pred) / y_test * 100
            mape = np.mean(pct_err)
            mean_acc = np.mean(np.maximum(0, 100 - pct_err))
            acc_10 = np.mean(pct_err <= 10.0) * 100
            acc_15 = np.mean(pct_err <= 15.0) * 100

            cv_mean = per_fold_df[name].mean()
            cv_std = per_fold_df[name].std()
            cv_min = per_fold_df[name].min()
            cv_max = per_fold_df[name].max()

            results.append({
                "Algorithm": name,
                "10-Fold Mean RMSE ($)": cv_mean,
                "10-Fold Std RMSE ($)": cv_std,
                "10-Fold Best ($)": cv_min,
                "10-Fold Worst ($)": cv_max,
                "Holdout R² (Accuracy)": h_r2,
                "Holdout Mean Accuracy (%)": mean_acc,
                "Holdout ±10% Accuracy (%)": acc_10,
                "Holdout ±15% Accuracy (%)": acc_15,
                "Holdout RMSE ($)": h_rmse,
                "Holdout MAE ($)": h_mae,
                "Holdout MAPE (%)": mape,
                "10-Fold Time (s)": cv_timing[name],
                "Holdout Train Time (s)": h_time,
            })

            print(f"{name:<24} | ${cv_mean:>11,.2f}  | {h_r2:>10.4f}  | ${h_rmse:>11,.2f}  | ${h_mae:>10,.2f}")

        print("-" * 95)

        summary_df = pd.DataFrame(results).sort_values("10-Fold Mean RMSE ($)").reset_index(drop=True)
        summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))
        self.summary_table = summary_df

        self.best_model_name = summary_df.iloc[0]["Algorithm"]
        print(f"\nWinning Algorithm by 10-Fold Cross-Validation: {self.best_model_name} (Rank 1)")

        # Save summary report
        summary_path = self.output_dir / "model_accuracy_summary.csv"
        self.summary_table.to_csv(summary_path, index=False)
        print(f"      Saved comprehensive summary to: {summary_path.name}")

        return self.summary_table

    def save_final_model(self, model_name=None):
        """
        Trains the winning algorithm on 100% of the dataset and serializes
        the final pipeline to 'salary_prediction_model.pkl'.
        """
        algo_name = model_name or self.best_model_name or "Lasso Regression"
        print(f"\n[5/6] Training Final Model ({algo_name}) on All 8,000 Records...")

        X = self.df_full[self.numeric_features + self.categorical_features]
        y = self.df_full[self.target_column]

        self.best_pipeline = Pipeline([
            ("preprocessor", self._build_preprocessor()),
            ("regressor", self.algorithms[algo_name])
        ])

        self.best_pipeline.fit(X, y)

        model_path = self.output_dir / "salary_prediction_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.best_pipeline, f)

        print(f"      Saved trained pipeline to: {model_path} ({os.path.getsize(model_path):,} bytes)")
        return self.best_pipeline

    def export_test_predictions(self):
        """
        Generates predictions on test_data.csv and saves test_predictions.csv
        and model_test_accuracy_comparison.csv.
        """
        if self.best_pipeline is None:
            self.save_final_model()

        print(f"\n[6/6] Generating Predictions on 1/3 Held-Out Test Set ({len(self.test_df):,} instances)...")
        X_test = self.test_df[self.numeric_features + self.categorical_features]
        y_test = self.test_df[self.target_column]

        preds = self.best_pipeline.predict(X_test)
        errors = y_test - preds
        abs_pct_errors = np.abs(errors) / y_test * 100
        accuracies = np.maximum(0, 100 - abs_pct_errors)

        comparison_df = self.test_df.copy()
        comparison_df["actual_salary_usd"] = y_test
        comparison_df["predicted_salary_usd"] = np.round(preds, 2)
        comparison_df["error_usd"] = np.round(errors, 2)
        comparison_df["absolute_percentage_error_%"] = np.round(abs_pct_errors, 2)
        comparison_df["individual_accuracy_%"] = np.round(accuracies, 2)
        comparison_df["accurate_within_10pct"] = abs_pct_errors <= 10.0
        comparison_df["accurate_within_15pct"] = abs_pct_errors <= 15.0

        cols_order = [
            "work_year", "job_title", "experience_level", "employment_type",
            "company_location", "employee_residence", "company_size", "remote_ratio",
            "actual_salary_usd", "predicted_salary_usd", "error_usd",
            "absolute_percentage_error_%", "individual_accuracy_%",
            "accurate_within_10pct", "accurate_within_15pct"
        ]
        comparison_df = comparison_df[cols_order]

        out_path1 = self.output_dir / "test_predictions.csv"
        out_path2 = self.output_dir / "model_test_accuracy_comparison.csv"

        comparison_df.to_csv(out_path1, index=False)
        comparison_df.to_csv(out_path2, index=False)

        print(f"      Saved test predictions to: {out_path1.name}")
        print(f"      Saved test accuracy file to: {out_path2.name}")

        return comparison_df

    def run_pipeline(self):
        """Runs complete training, 10-fold cross validation, and export pipeline."""
        print("=" * 90)
        print("AI & DATA SCIENCE SALARY PREDICTOR - 10-FOLD CV MODEL TRAINER & SPLITTER")
        print("=" * 90)
        self.load_and_split_data(test_size=1/3)
        self.train_and_evaluate()
        self.save_final_model()
        self.export_test_predictions()
        print("\n" + "=" * 90)
        print("10-FOLD CV TRAINING PIPELINE COMPLETE! ALL FILES GENERATED.")
        print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description="Train AI/DS Salary models with 10-Fold CV")
    parser.add_argument("--data", type=str, default=None, help="Path to raw csv dataset")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds")
    args = parser.parse_args()

    trainer = SalaryModelTrainer(data_path=args.data, output_dir=args.out, n_splits=args.folds)
    trainer.run_pipeline()

if __name__ == "__main__":
    main()
