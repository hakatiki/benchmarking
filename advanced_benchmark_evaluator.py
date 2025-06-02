import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import skew, kurtosis as scipy_kurtosis
import argparse
from pathlib import Path
import re

# ANSI escape codes for console colors
class ConsoleColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def cprint(color, *args):
    """Prints text in specified color."""
    print(f"{color}{' '.join(map(str, args))}{ConsoleColors.ENDC}")

class ModelBenchmarkEvaluator:
    def __init__(self, results_dir: str = "results", plots_dir: str = "plots_per_model", config_color_mode: bool = True):
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.all_benchmark_files = []
        self.config_color_mode = config_color_mode # If true, use user's specific color names

        # Color scheme from user
        self.palette = {
            "Blue_20": "#D1E4F1", "Blue_40": "#A3CAE3", "Blue_80": "#4695C8", "Blue": "#187ABA",
            "Navy_20": "#CCD5DC", "Navy_40": "#99AAB9", "Navy_80": "#335574", "Navy": "#002B51",
        }
        # General plotting colors, can be overridden by model or type
        self.general_colors = {
            'api': self.palette["Blue"],
            'manual': self.palette["Navy"],
            'primary': self.palette["Blue_80"],
            'secondary': self.palette["Navy_80"],
            'error': '#FF6347' # Tomato
        }

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self._setup_matplotlib_style()
        cprint(ConsoleColors.HEADER, "📊 Model Performance Evaluator Initialized 📊")

    def _setup_matplotlib_style(self):
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('seaborn-whitegrid') # Fallback
        plt.rcParams.update({
            'figure.figsize': (12, 7), 'axes.titlesize': 16, 'axes.labelsize': 12,
            'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
            'savefig.dpi': 300, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'
        })

    def _parse_filename(self, filepath: Path):
        filename = filepath.name
        # Pattern: (api_benchmark_|manual_)(MODEL_NAME)_(YYYYMMDD_HHMMSS).json
        # Work backwards from the timestamp pattern to handle model names with underscores
        match = re.match(r"^(api_benchmark_|manual_)(.+)_(\d{8}_\d{6})\.json$", filename)
        
        if match:
            prefix = match.group(1)
            model_name = match.group(2)
            timestamp_str = match.group(3)
            run_type = 'api' if prefix == 'api_benchmark_' else 'manual'
            try:
                datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S') # Validate timestamp
                return {"type": run_type, "model_name": model_name, "timestamp_str": timestamp_str, "file_path": filepath, "original_filename": filename}
            except ValueError:
                cprint(ConsoleColors.WARNING, f"Invalid timestamp in filename: {filename}")
        else:
            cprint(ConsoleColors.WARNING, f"Could not parse filename: {filename}")
        return None

    def load_benchmark_files(self):
        cprint(ConsoleColors.OKBLUE, f"🔍 Loading benchmark files from: {self.results_dir}")
        if not self.results_dir.exists():
            cprint(ConsoleColors.FAIL, f"❌ Results directory not found: {self.results_dir}")
            return False

        self.all_benchmark_files = []
        for f_path in self.results_dir.glob("*.json"):
            parsed_info = self._parse_filename(f_path)
            if parsed_info:
                self.all_benchmark_files.append(parsed_info)
        
        self.all_benchmark_files.sort(key=lambda x: (x["model_name"], x["type"], x["timestamp_str"]))
        cprint(ConsoleColors.OKGREEN, f"  Found {len(self.all_benchmark_files)} processable benchmark files.")
        return True

    def _get_descriptive_stats(self, data_series: pd.Series, precision: int = 4) -> dict:
        if data_series.empty or data_series.isnull().all():
            return {stat: np.nan for stat in ['count', 'mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis']}
        
        numeric_series = pd.to_numeric(data_series.dropna(), errors='coerce').dropna()
        if numeric_series.empty:
            return {stat: np.nan for stat in ['count', 'mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis']}

        stats = {
            'count': int(numeric_series.count()),
            'mean': numeric_series.mean(),
            'median': numeric_series.median(),
            'std': numeric_series.std(),
            'min': numeric_series.min(),
            'max': numeric_series.max(),
            'skewness': skew(numeric_series, nan_policy='omit') if len(numeric_series) > 1 else np.nan,
            'kurtosis': scipy_kurtosis(numeric_series, fisher=True, nan_policy='omit') if len(numeric_series) > 3 else np.nan,
        }
        return {k: (round(v, precision) if isinstance(v, (float, np.floating)) and not np.isnan(v) else v) for k, v in stats.items()}

    def _generate_run_report_text(self, df_run: pd.DataFrame, file_info: dict, run_metadata: dict) -> str:
        report_lines = [
            f"PERFORMANCE REPORT for {file_info['model_name']} ({file_info['type']} run)",
            "=" * 60,
            f"Source File: {file_info['original_filename']}",
            f"Run Type: {file_info['type'].upper()}",
            f"Model: {run_metadata.get('model', file_info['model_name'])}", # Use metadata if more specific
            f"Timestamp (from filename): {file_info['timestamp_str']}",
            f"Total API Calls in this run: {len(df_run)} (excluding errors)",
            f"Run Duration (metadata): {run_metadata.get('total_duration_seconds', 0):.2f}s\n"
        ]

        metrics_to_analyze = {
            "Response Time (s)": "response_time_seconds",
            "Tokens Per Second (TPS)": "tokens_per_second",
            "Prompt Tokens": "prompt_tokens",
            "Completion Tokens": "completion_tokens",
            "Total Tokens": "total_tokens",
            "Cost per Call (USD)": "cost_usd.total_cost" # Path for nested cost
        }

        for display_name, col_name_path in metrics_to_analyze.items():
            report_lines.append(f"{display_name} Statistics:")
            # Handle potentially nested column names like 'cost_usd.total_cost'
            data_to_stat = df_run
            if '.' in col_name_path:
                # This assumes cost_usd is a dict in each row of results
                # We need to extract it properly before creating the DataFrame or handle it here
                # For simplicity, let's assume _create_run_dataframe handles this extraction
                # and creates a flat 'total_cost_usd' column
                actual_col_name = col_name_path.split('.')[-1] # e.g. total_cost
                # This part needs robust handling in df creation. Assuming df_run has flat columns now.
                # For the report, we'll use the direct column name we create later.
                # This dictionary is more for conceptual mapping for now.
                # The actual column used for stats lookup will be what _create_run_dataframe produces.
                # e.g. if col_name_path is cost_usd.total_cost, _create_run_dataframe should make a 'total_cost_usd' column
                
                # Let's adjust the conceptual col_name for stats lookup
                if col_name_path == "cost_usd.total_cost": actual_col_name = "total_cost_usd"
                else: actual_col_name = col_name_path

            else:
                actual_col_name = col_name_path
            
            if actual_col_name in df_run.columns:
                stats = self._get_descriptive_stats(df_run[actual_col_name])
                report_lines.append(f"  Count: {stats['count']}, Mean: {stats['mean']}, Median: {stats['median']}, Std: {stats['std']}")
                report_lines.append(f"  Min: {stats['min']}, Max: {stats['max']}, Skew: {stats['skewness']}, Kurt: {stats['kurtosis']}\n")
            else:
                report_lines.append(f"  Data for '{display_name}' (column: {actual_col_name}) not found in DataFrame.\n")


        if file_info['type'] == 'api' and 'cached_tokens' in df_run.columns:
            total_prompt_tokens = df_run['prompt_tokens'].sum()
            total_cached_tokens = df_run['cached_tokens'].sum()
            cache_hit_rate = (total_cached_tokens / total_prompt_tokens * 100) if total_prompt_tokens > 0 else 0
            report_lines.append("API Cache Performance:")
            report_lines.append(f"  Total Prompt Tokens in run: {total_prompt_tokens:.0f}")
            report_lines.append(f"  Total Cached Tokens in run: {total_cached_tokens:.0f}")
            report_lines.append(f"  Overall Cache Hit Rate for this run: {cache_hit_rate:.2f}%\n")
            # Could also report on savings if that data is consistently in api_benchmark results

        return "\n".join(report_lines)

    def _plot_metric_distribution(self, data_series: pd.Series, title: str, xlabel: str, output_path: Path, color: str):
        if data_series.empty or data_series.isnull().all():
            cprint(ConsoleColors.WARNING, f"Skipping plot '{title}' due to empty or all-NaN data.")
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(data_series, kde=True, color=color, bins=20, stat="density") # Use density for y-axis
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density") # Changed from Frequency to Density
        
        mean_val = data_series.mean()
        median_val = data_series.median()
        if pd.notna(mean_val):
            plt.axvline(mean_val, color=self.general_colors['error'], linestyle='--', label=f"Mean: {mean_val:.2f}")
        if pd.notna(median_val):
            plt.axvline(median_val, color='purple', linestyle=':', label=f"Median: {median_val:.2f}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path.with_name(f"{output_path.name}_hist.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.boxplot(y=data_series, color=color)
        plt.title(f"Box Plot of {title}")
        plt.ylabel(xlabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path.with_name(f"{output_path.name}_boxplot.png"))
        plt.close()

    def _create_run_dataframe(self, results_list: list) -> pd.DataFrame:
        """Creates a DataFrame from the 'results' list of a single benchmark file."""
        if not results_list:
            return pd.DataFrame()

        # Flatten cost data and handle potential errors/missing keys gracefully
        processed_results = []
        for call_result in results_list:
            if 'error' in call_result:
                continue # Skip errored calls for performance df

            flat_result = call_result.copy() # Start with a copy
            cost_data = flat_result.pop('cost_usd', {}) # Remove and get cost_usd dict
            
            flat_result['total_cost_usd'] = cost_data.get('total_cost', np.nan)
            # Add other cost components if needed for detailed analysis later
            # flat_result['input_cost_usd'] = cost_data.get('input_cost', np.nan)
            # flat_result['output_cost_usd'] = cost_data.get('output_cost', np.nan)
            
            # Ensure other key numeric fields exist, defaulting to NaN if not
            numeric_fields = ['prompt_tokens', 'completion_tokens', 'total_tokens', 
                              'response_time_seconds', 'tokens_per_second', 'cached_tokens']
            for field in numeric_fields:
                if field not in flat_result:
                    flat_result[field] = np.nan
            if 'cached_tokens' not in flat_result: # Ensure it exists even if type != api for schema consistency
                 flat_result['cached_tokens'] = 0


            processed_results.append(flat_result)
        
        df = pd.DataFrame(processed_results)
        
        # Convert relevant columns to numeric, coercing errors
        cols_to_numeric = ['prompt_tokens', 'completion_tokens', 'total_tokens', 
                           'response_time_seconds', 'tokens_per_second', 
                           'total_cost_usd', 'cached_tokens']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def process_single_benchmark_file(self, file_info: dict) -> dict:
        """Processes a single benchmark file for its individual performance characteristics."""
        original_filename = file_info["original_filename"]
        model_name = file_info["model_name"]
        run_type = file_info["type"]

        # Create subdirectory for this file's results
        # Replace potential problematic characters in filename for dir name
        safe_filename_base = original_filename.replace('.json', '').replace('.', '_')
        run_output_dir = self.plots_dir / f"{safe_filename_base}_eval"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        cprint(ConsoleColors.OKCYAN, f"Processing File: {original_filename} (Model: {model_name}, Type: {run_type.upper()})")
        cprint(ConsoleColors.OKCYAN, f"  Outputting to: {run_output_dir}")

        try:
            with open(file_info["file_path"], 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception as e:
            cprint(ConsoleColors.FAIL, f"  Error loading JSON from {original_filename}: {e}")
            return None

        results_list = raw_data.get("results", [])
        run_metadata = raw_data.get("benchmark_info", {})

        if not results_list:
            cprint(ConsoleColors.WARNING, f"  No 'results' data in {original_filename}. Skipping.")
            return None

        df_run = self._create_run_dataframe(results_list)
        if df_run.empty:
            cprint(ConsoleColors.WARNING, f"  DataFrame empty after processing results from {original_filename} (all calls might have errored or missing data). Skipping.")
            return None

        # Generate and save text report
        report_content = self._generate_run_report_text(df_run, file_info, run_metadata)
        with open(run_output_dir / "run_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_content)
        cprint(ConsoleColors.OKGREEN, f"    Run report saved: {run_output_dir / 'run_report.txt'}")

        # Plotting
        plot_color = self.general_colors['api'] if run_type == 'api' else self.general_colors['manual']
        
        metrics_to_plot = {
            "Response Time (s)": ("response_time_seconds", "s"),
            "Tokens Per Second (TPS)": ("tokens_per_second", "Tokens/s"),
            "Prompt Tokens": ("prompt_tokens", "Tokens"),
            "Completion Tokens": ("completion_tokens", "Tokens"),
            "Total Cost per Call (USD)": ("total_cost_usd", "USD")
        }
        for display_name, (col_name, unit) in metrics_to_plot.items():
            if col_name in df_run.columns:
                safe_col_name_for_file = col_name.replace('.', '_') # for filename
                self._plot_metric_distribution(
                    df_run[col_name],
                    title=f"{display_name} - {model_name} ({run_type.upper()})",
                    xlabel=f"{display_name}", # Unit is already in display name
                    output_path=run_output_dir / f"{safe_col_name_for_file}_distribution",
                    color=plot_color
                )
            else:
                 cprint(ConsoleColors.WARNING, f"    Column '{col_name}' not found for plotting '{display_name}'.")


        cprint(ConsoleColors.OKGREEN, f"    Run-specific plots saved to {run_output_dir}")
        
        # Collect key aggregates for overall summary
        # Ensure all these stats are actually calculated and present
        summary_data = {
            "original_filename": original_filename,
            "model_name": model_name,
            "run_type": run_type,
            "timestamp_str": file_info["timestamp_str"],
            "num_calls": len(df_run),
            "mean_response_time_s": df_run['response_time_seconds'].mean() if 'response_time_seconds' in df_run else np.nan,
            "median_response_time_s": df_run['response_time_seconds'].median() if 'response_time_seconds' in df_run else np.nan,
            "std_response_time_s": df_run['response_time_seconds'].std() if 'response_time_seconds' in df_run else np.nan,
            "mean_tps": df_run['tokens_per_second'].mean() if 'tokens_per_second' in df_run else np.nan,
            "median_tps": df_run['tokens_per_second'].median() if 'tokens_per_second' in df_run else np.nan,
            "std_tps": df_run['tokens_per_second'].std() if 'tokens_per_second' in df_run else np.nan,
            "mean_prompt_tokens": df_run['prompt_tokens'].mean() if 'prompt_tokens' in df_run else np.nan,
            "mean_completion_tokens": df_run['completion_tokens'].mean() if 'completion_tokens' in df_run else np.nan,
            "mean_total_cost_usd": df_run['total_cost_usd'].mean() if 'total_cost_usd' in df_run else np.nan,
            "total_cost_sum_usd": df_run['total_cost_usd'].sum() if 'total_cost_usd' in df_run else np.nan,
            "api_cache_hit_rate": (df_run['cached_tokens'].sum() / df_run['prompt_tokens'].sum() * 100) \
                                  if run_type == 'api' and 'cached_tokens' in df_run and 'prompt_tokens' in df_run and df_run['prompt_tokens'].sum() > 0 \
                                  else np.nan
        }
        return summary_data


    def generate_overall_analysis(self, all_run_summaries: list):
        if not all_run_summaries:
            cprint(ConsoleColors.WARNING, "No run summaries available to generate overall analysis.")
            return

        cprint(ConsoleColors.OKBLUE, "📊 Generating Overall Analysis Report and Plots...")
        overall_df = pd.DataFrame([s for s in all_run_summaries if s is not None]) # Filter out None entries

        if overall_df.empty:
            cprint(ConsoleColors.WARNING, "DataFrame for overall analysis is empty. No data to summarize.")
            return

        summary_report_path = self.plots_dir / "overall_analysis_report.txt"
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            f.write("OVERALL BENCHMARK ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total Benchmark Files Processed: {len(all_run_summaries)}\n")
            f.write(f"Unique Models Found: {', '.join(overall_df['model_name'].unique())}\n\n")

            f.write("Summary of Processed Benchmark Runs:\n")
            for _, row in overall_df.iterrows():
                f.write(f"  - File: {row['original_filename']} (Model: {row['model_name']}, Type: {row['run_type'].upper()})\n")
                f.write(f"    Mean TPS: {row['mean_tps']:.2f}, Mean Resp. Time: {row['mean_response_time_s']:.2f}s, Mean Cost/Call: ${row['mean_total_cost_usd']:.6f}\n")
                if pd.notna(row['api_cache_hit_rate']):
                     f.write(f"    API Cache Hit Rate: {row['api_cache_hit_rate']:.2f}%\n")
                f.write("\n")
            
            f.write("\nModel-Level Aggregated Performance (Averages of Run Averages):\n")
            # Group by model and type, then average the means (or other relevant aggregates)
            # This is an average of averages, which can be skewed if runs have different numbers of calls.
            # A weighted average or concatenating all raw results first might be more robust for some metrics.
            # For now, simple average of run means.
            model_summary = overall_df.groupby(['model_name', 'run_type']).agg(
                avg_mean_tps=('mean_tps', 'mean'),
                avg_median_tps=('median_tps', 'mean'),
                avg_mean_response_time_s=('mean_response_time_s', 'mean'),
                avg_median_response_time_s=('median_response_time_s', 'mean'),
                avg_mean_total_cost_usd=('mean_total_cost_usd', 'mean'),
                avg_api_cache_hit_rate=('api_cache_hit_rate', 'mean'), # Avg of hit rates
                num_runs=('original_filename', 'count')
            ).reset_index()

            for _, row in model_summary.iterrows():
                f.write(f"Model: {row['model_name']}, Type: {row['run_type'].upper()} ({row['num_runs']} run(s))\n")
                f.write(f"  Avg. Mean TPS: {row['avg_mean_tps']:.2f}\n")
                f.write(f"  Avg. Median TPS: {row['avg_median_tps']:.2f}\n")
                f.write(f"  Avg. Mean Response Time: {row['avg_mean_response_time_s']:.2f}s\n")
                f.write(f"  Avg. Median Response Time: {row['avg_median_response_time_s']:.2f}s\n")
                f.write(f"  Avg. Mean Cost/Call: ${row['avg_mean_total_cost_usd']:.6f}\n")
                if pd.notna(row['avg_api_cache_hit_rate']):
                     f.write(f"  Avg. API Cache Hit Rate: {row['avg_api_cache_hit_rate']:.2f}%\n")
                f.write("\n")

        cprint(ConsoleColors.OKGREEN, f"Overall analysis report saved: {summary_report_path}")

        # Overall Comparative Plots
        # Boxplot of Mean TPS by Model (distinguishing run_type)
        if 'mean_tps' in overall_df.columns and not overall_df['mean_tps'].isnull().all():
            plt.figure(figsize=(max(12, len(overall_df['model_name'].unique()) * 1.5), 7))
            sns.boxplot(x='model_name', y='mean_tps', hue='run_type', data=overall_df, palette={'api': self.palette['Blue'], 'manual': self.palette['Navy']})
            plt.title('Overall Mean Tokens Per Second (TPS) by Model and Run Type')
            plt.ylabel('Mean TPS (Average of Run Means)')
            plt.xlabel('Model')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Run Type')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "overall_mean_tps_by_model.png")
            plt.close()

        if 'mean_response_time_s' in overall_df.columns and not overall_df['mean_response_time_s'].isnull().all():
            plt.figure(figsize=(max(12, len(overall_df['model_name'].unique()) * 1.5), 7))
            sns.boxplot(x='model_name', y='mean_response_time_s', hue='run_type', data=overall_df, palette={'api': self.palette['Blue_80'], 'manual': self.palette['Navy_80']})
            plt.title('Overall Mean Response Time (s) by Model and Run Type')
            plt.ylabel('Mean Response Time (s) (Average of Run Means)')
            plt.xlabel('Model')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Run Type')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "overall_mean_response_time_by_model.png")
            plt.close()
        
        if 'mean_total_cost_usd' in overall_df.columns and not overall_df['mean_total_cost_usd'].isnull().all():
            plt.figure(figsize=(max(12, len(overall_df['model_name'].unique()) * 1.5), 7))
            # Bar plot might be better for mean costs
            sns.barplot(x='model_name', y='mean_total_cost_usd', hue='run_type', data=overall_df, palette={'api': self.palette['Blue_40'], 'manual': self.palette['Navy_40']}, estimator=np.mean, errorbar='sd')
            plt.title('Overall Mean Cost per Call (USD) by Model and Run Type')
            plt.ylabel('Mean Cost per Call (USD) (Average of Run Means)')
            plt.xlabel('Model')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Run Type')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "overall_mean_cost_per_call_by_model.png")
            plt.close()

        cprint(ConsoleColors.OKGREEN, f"Overall comparative plots saved to {self.plots_dir}")


    def run_evaluation(self):
        cprint(ConsoleColors.HEADER, "🚀 Starting Model Performance Evaluation Suite 🚀")
        if not self.load_benchmark_files():
            return # Stop if results dir not found

        if not self.all_benchmark_files:
            cprint(ConsoleColors.WARNING, "No benchmark files found or parsed. Evaluation cannot proceed.")
            return

        all_run_summaries = []
        for file_info in self.all_benchmark_files:
            try:
                summary = self.process_single_benchmark_file(file_info)
                if summary:
                    all_run_summaries.append(summary)
            except Exception as e:
                cprint(ConsoleColors.FAIL, f"  UNHANDLED Error processing file {file_info.get('original_filename', 'N/A')}: {e}")
                import traceback
                traceback.print_exc()
        
        if all_run_summaries:
            self.generate_overall_analysis(all_run_summaries)
        else:
            cprint(ConsoleColors.WARNING, "No data was successfully processed from any benchmark files for the overall analysis.")

        cprint(ConsoleColors.HEADER, "🎉 Model Performance Evaluation Complete! 🎉")
        cprint(ConsoleColors.OKBLUE, f"Check the '{self.plots_dir}' directory for reports and plots.")

def main():
    parser = argparse.ArgumentParser(description="Evaluates performance characteristics of models from benchmark JSON files.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing benchmark JSON files.")
    parser.add_argument("--plots-dir", type=str, default="plots_model_eval", help="Main directory to save evaluation plots and reports.")
    # config_color_mode can be added if you want to switch between user's specific color names and generic ones
    args = parser.parse_args()

    evaluator = ModelBenchmarkEvaluator(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir
    )
    evaluator.run_evaluation()

if __name__ == "__main__":
    main() 