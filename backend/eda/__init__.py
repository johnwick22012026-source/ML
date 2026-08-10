"""EDA service providing lightweight data summaries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["EDAService"]


class EDAService:
    """Entry point for generating exploratory data analysis artifacts."""

    VALID_CORRELATION_METHODS = {"pearson", "spearman", "kendall"}

    def analyze(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns structured exploratory analysis artifacts driven by configuration."""
        df = self._ensure_dataframe(data)
        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        hist_columns = self._resolve_columns(config.get("histogram", {}).get("columns"), numeric_columns)
        kde_columns = self._resolve_columns(config.get("kde", {}).get("columns"), numeric_columns)
        box_columns = self._resolve_columns(config.get("boxplot", {}).get("columns"), numeric_columns)
        scatter_pairs = self._resolve_pairs(config.get("scatter", {}).get("pairs"), numeric_columns)
        pair_plot_columns = self._resolve_columns(config.get("pair_plot", {}).get("columns"), numeric_columns)
        correlation_method = self._resolve_correlation_method(
            config.get("correlation", {}).get("method")
        )
        correlation_columns = self._resolve_columns(config.get("correlation", {}).get("columns"), numeric_columns)

        numeric_summary = self._build_numeric_summary(df)
        missing_summary = self._build_missing_value_summary(df)
        histograms = self._build_histograms(df, hist_columns)
        kdes = self._build_kdes(df, kde_columns)
        boxplots = self._build_boxplots(df, box_columns)
        scatterplots = self._build_scatterplots(df, scatter_pairs)
        correlation_matrix = self._build_correlation(df, correlation_columns, correlation_method)
        pair_plot_data = self._build_pair_plot_data(df, pair_plot_columns)
        outlier_summary = self._build_outlier_summary(df, box_columns)

        return {
            "numerical_summary": numeric_summary,
            "missing_value_summary": missing_summary,
            "histograms": histograms,
            "kde": kdes,
            "boxplots": boxplots,
            "scatterplots": scatterplots,
            "correlation_matrix": correlation_matrix,
            "pair_plot_data": pair_plot_data,
            "outlier_summary": outlier_summary,
            "config": config,
        }

    def _ensure_dataframe(self, data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise ValueError("Unsupported data type for EDA service")

    def _resolve_columns(self, configured: Optional[Sequence[str]], defaults: List[str]) -> List[str]:
        if configured:
            return [c for c in configured if c in defaults]
        return defaults

    def _resolve_pairs(self, configured: Optional[Sequence[Tuple[str, str]]], defaults: List[str]) -> List[Tuple[str, str]]:
        if configured:
            return [pair for pair in configured if pair[0] in defaults and pair[1] in defaults]
        pairs: List[Tuple[str, str]] = []
        if len(defaults) >= 2:
            for i in range(min(3, len(defaults) - 1)):
                pairs.append((defaults[i], defaults[i + 1]))
        return pairs

    def _resolve_correlation_method(self, method: Optional[str]) -> str:
        if not method:
            return "pearson"
        normalized = method.lower()
        if normalized not in self.VALID_CORRELATION_METHODS:
            valid_methods = ", ".join(sorted(self.VALID_CORRELATION_METHODS))
            raise ValueError(
                f"Unsupported correlation method '{method}'. Valid methods are: {valid_methods}."
            )
        return normalized

    def _build_numeric_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        summary = df.describe(include="number").to_dict()
        counts = {
            col: {
                "count": int(df[col].count()),
                "unique": int(df[col].nunique(dropna=True)),
                "dtype": str(df[col].dtype),
            }
            for col in df.columns
        }
        return {"descriptive": summary, "counts": counts}

    def _build_missing_value_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        total_cells = int(df.shape[0] * df.shape[1])
        missing = df.isna().sum()
        missing_percent = (missing / df.shape[0] * 100).round(2)
        return {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "total_missing": int(missing.sum()),
            "missing_percent": round(missing.sum() / total_cells * 100, 2) if total_cells > 0 else 0.0,
            "per_column": {
                col: {
                    "missing_count": int(missing[col]),
                    "missing_percent": float(missing_percent[col]),
                    "dtype": str(df[col].dtype),
                }
                for col in df.columns
            },
        }

    def _build_histograms(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, List[float]]]:
        histograms = {}
        for col in columns:
            series = df[col].dropna()
            if series.empty:
                continue
            counts, bin_edges = np.histogram(series, bins=10)
            histograms[col] = {
                "bins": bin_edges.tolist(),
                "counts": counts.astype(int).tolist(),
            }
        return histograms

    def _build_kdes(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, List[float]]]:
        kdes: Dict[str, Dict[str, List[float]]] = {}
        for col in columns:
            series = df[col].dropna().astype(float)
            if series.nunique() < 2:
                continue
            x_vals = np.linspace(series.min(), series.max(), 200)
            bandwidth = self._select_bandwidth(series)
            kde_vals = self._compute_kde(series.values, x_vals, bandwidth)
            kdes[col] = {
                "x": x_vals.tolist(),
                "y": kde_vals.tolist(),
            }
        return kdes

    def _select_bandwidth(self, values: pd.Series) -> float:
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        std = float(values.std())
        n = max(len(values), 1)
        return max(1e-3, 0.9 * min(std, iqr / 1.349) * n ** (-1 / 5))

    def _compute_kde(self, sample: np.ndarray, points: np.ndarray, bandwidth: float) -> np.ndarray:
        if sample.size == 0:
            return np.zeros_like(points)
        kernel_values = np.exp(-0.5 * ((points[:, None] - sample[None, :]) / bandwidth) ** 2)
        kernel_values = kernel_values / (bandwidth * np.sqrt(2 * np.pi))
        densities = kernel_values.mean(axis=1)
        densities = densities / (densities.max() or 1)
        return densities

    def _build_boxplots(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
        boxplots = {}
        for col in columns:
            series = df[col].dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            median = float(series.quantile(0.5))
            iqr = q3 - q1
            lower_whisker = float(max(series.min(), q1 - 1.5 * iqr))
            upper_whisker = float(min(series.max(), q3 + 1.5 * iqr))
            outliers = series[(series < lower_whisker) | (series > upper_whisker)].tolist()
            boxplots[col] = {
                "lower_whisker": lower_whisker,
                "q1": q1,
                "median": median,
                "q3": q3,
                "upper_whisker": upper_whisker,
                "outliers": outliers[:20],
            }
        return boxplots

    def _build_scatterplots(
        self, df: pd.DataFrame, pairs: List[Tuple[str, str]]
    ) -> Dict[str, Dict[str, List[Any]]]:
        scatterplots: Dict[str, Dict[str, List[Any]]] = {}
        for x_col, y_col in pairs:
            series_x = df[x_col].dropna()
            series_y = df[y_col].dropna()
            aligned = pd.concat([series_x, series_y], axis=1).dropna()
            if aligned.empty:
                continue
            key = f"{x_col}_vs_{y_col}"
            scatterplots[key] = {
                "x": aligned[x_col].tolist(),
                "y": aligned[y_col].tolist(),
                "x_label": x_col,
                "y_label": y_col,
            }
        return scatterplots

    def _build_correlation(
        self, df: pd.DataFrame, columns: List[str], method: str
    ) -> Dict[str, Any]:
        if not columns:
            return {"method": method, "matrix": {}, "columns": []}
        corr = df[columns].corr(method=method)
        return {
            "method": method,
            "columns": corr.columns.tolist(),
            "matrix": corr.round(4).values.tolist(),
        }

    def _build_pair_plot_data(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, List[Any]]:
        if not columns:
            return {}
        limited = columns[:4]
        return {
            "columns": limited,
            "data": df[limited].dropna().to_dict(orient="list"),
        }

    def _build_outlier_summary(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for col in columns:
            series = df[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            lower = float(q1 - 1.5 * iqr)
            upper = float(q3 + 1.5 * iqr)
            outliers = series[(series < lower) | (series > upper)]
            summary[col] = {
                "outlier_count": int(outliers.count()),
                "outlier_ratio": round(outliers.count() / max(series.count(), 1), 4),
                "sample_outliers": outliers.head(10).tolist(),
                "bounds": {"lower": lower, "upper": upper},
            }
        return summary
