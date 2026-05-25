#!/usr/bin/env python3
"""AutoML Pipeline Orchestrator - End-to-end automated ML."""
import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


@dataclass
class ExperimentResult:
    model_name: str
    params: Dict
    cv_score: float
    test_r2: float
    test_rmse: float
    training_time: float


class DataProfiler:
    """Profile dataset characteristics."""
    
    def profile(self, df: pd.DataFrame, target: str) -> Dict:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        profile = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_features": [c for c in numeric_cols if c != target],
            "categorical_features": categorical_cols,
            "target_stats": df[target].describe().to_dict() if target in numeric_cols else {"type": "categorical", "classes": df[target].nunique()},
            "missing_values": df.isnull().sum().to_dict(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        return profile


class FeatureEngineer:
    """Generate and select features."""
    
    def engineer(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        df = df.copy()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create interaction features for top correlated pairs
        if len(numeric) > 2:
            corr = df[numeric].corrwith(df[target]).abs().sort_values(ascending=False)
            top_features = corr.head(3).index.tolist()
            if target in top_features:
                top_features.remove(target)
            
            for i in range(min(2, len(top_features))):
                for j in range(i+1, min(3, len(top_features))):
                    col_name = f"{top_features[i]}_x_{top_features[j]}"
                    df[col_name] = df[top_features[i]] * df[top_features[j]]
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown", inplace=True)
        
        return df


class ModelSelector:
    """Train and compare multiple models."""
    
    def __init__(self, time_limit: int = 3600):
        self.time_limit = time_limit
        self.models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0)
        }
    
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[ExperimentResult]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results = []
        
        for name, model in self.models.items():
            start = time.time()
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Final training
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            elapsed = time.time() - start
            
            results.append(ExperimentResult(
                model_name=name,
                params=model.get_params(),
                cv_score=cv_scores.mean(),
                test_r2=r2_score(y_test, y_pred),
                test_rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
                training_time=elapsed
            ))
        
        return sorted(results, key=lambda x: x.test_r2, reverse=True)


class AutoMLOrchestrator:
    """Main orchestrator."""
    
    def __init__(self, time_limit: int = 3600):
        self.profiler = DataProfiler()
        self.engineer = FeatureEngineer()
        self.selector = ModelSelector(time_limit)
    
    def run(self, data_path: str, target: str) -> Dict:
        print(f"[📊] Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        print("[🔍] Profiling dataset...")
        profile = self.profiler.profile(df, target)
        print(f"    Rows: {profile['rows']}, Features: {profile['columns']-1}")
        
        print("[🔧] Engineering features...")
        df_engineered = self.engineer.engineer(df, target)
        new_features = len(df_engineered.columns) - len(df.columns)
        print(f"    Generated {new_features} new features")
        
        # Prepare features
        X = df_engineered.drop(columns=[target])
        y = df_engineered[target]
        
        # Encode categoricals
        X = pd.get_dummies(X, drop_first=True)
        
        print("[🤖] Training models...")
        results = self.selector.select(X, y)
        
        best = results[0]
        return {
            "dataset_profile": profile,
            "features_engineered": new_features,
            "best_model": best.model_name,
            "best_r2": best.test_r2,
            "best_rmse": best.test_rmse,
            "all_results": [
                {
                    "model": r.model_name,
                    "cv_r2": r.cv_score,
                    "test_r2": r.test_r2,
                    "test_rmse": r.test_rmse,
                    "time_sec": r.training_time
                }
                for r in results
            ],
            "recommendation": f"Deploy {best.model_name} (R2={best.test_r2:.3f})"
        }


def main():
    parser = argparse.ArgumentParser(description="AutoML Pipeline")
    parser.add_argument("--data", required=True, help="CSV file path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--output", default="automl_report.json")
    args = parser.parse_args()
    
    orchestrator = AutoMLOrchestrator(args.time_limit)
    result = orchestrator.run(args.data, args.target)
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[✓] Best model: {result['best_model']}")
    print(f"    Test R2: {result['best_r2']:.4f}")
    print(f"    Report: {args.output}")


if __name__ == "__main__":
    main()
