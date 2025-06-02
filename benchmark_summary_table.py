import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import re

def parse_filename(filepath: Path):
    """Parse benchmark filename to extract model and type info."""
    filename = filepath.name
    match = re.match(r"^(api_benchmark_|manual_)(.+)_(\d{8}_\d{6})\.json$", filename)
    
    if match:
        prefix = match.group(1)
        model_name = match.group(2)
        timestamp_str = match.group(3)
        run_type = 'API' if prefix == 'api_benchmark_' else 'Manual'
        try:
            datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            return {"type": run_type, "model": model_name, "timestamp": timestamp_str}
        except ValueError:
            pass
    return None

def load_benchmark_summary(filepath: Path):
    """Load a single benchmark file and extract summary metrics."""
    file_info = parse_filename(filepath)
    if not file_info:
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath.name}: {e}")
        return None
    
    results = data.get("results", [])
    metadata = data.get("benchmark_info", {})
    
    if not results:
        return None
    
    # Filter out errored calls
    successful_calls = [r for r in results if 'error' not in r]
    
    if not successful_calls:
        return None
    
    # Extract metrics
    total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in successful_calls)
    total_completion_tokens = sum(r.get('completion_tokens', 0) for r in successful_calls)
    total_tokens = sum(r.get('total_tokens', 0) for r in successful_calls)
    
    # Calculate averages
    response_times = [r.get('response_time_seconds', 0) for r in successful_calls]
    tps_values = [r.get('tokens_per_second', 0) for r in successful_calls]
    costs = [r.get('cost_usd', {}).get('total_cost', 0) for r in successful_calls]
    
    avg_response_time = np.mean(response_times) if response_times else 0
    avg_tps = np.mean(tps_values) if tps_values else 0
    avg_cost_per_call = np.mean(costs) if costs else 0
    total_cost = sum(costs) if costs else 0
    
    # Cache metrics (for API runs)
    cached_tokens = 0
    cache_hit_rate = 0
    if file_info['type'] == 'API':
        cached_tokens = sum(r.get('cached_tokens', 0) for r in successful_calls)
        cache_hit_rate = (cached_tokens / total_prompt_tokens * 100) if total_prompt_tokens > 0 else 0
    
    return {
        'Filename': filepath.name,
        'Model': file_info['model'],
        'Type': file_info['type'],
        'Timestamp': file_info['timestamp'],
        'Num_Calls': len(successful_calls),
        'Total_Prompt_Tokens': total_prompt_tokens,
        'Total_Completion_Tokens': total_completion_tokens,
        'Total_Tokens': total_tokens,
        'Avg_Response_Time_s': round(avg_response_time, 3),
        'Avg_TPS': round(avg_tps, 2),
        'Avg_Cost_per_Call_USD': round(avg_cost_per_call, 6),
        'Total_Cost_USD': round(total_cost, 6),
        'Cached_Tokens': cached_tokens,
        'Cache_Hit_Rate_%': round(cache_hit_rate, 2),
        'Duration_s': metadata.get('total_duration_seconds', 0),
        'Start_Time': metadata.get('start_time', 'N/A'),
    }

def create_benchmark_summary_table(results_dir: str = "results", output_file: str = None):
    """Create a comprehensive summary table of all benchmark results."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    print(f"🔍 Loading benchmark files from: {results_dir}")
    
    # Load all benchmark files
    summaries = []
    for json_file in results_path.glob("*.json"):
        summary = load_benchmark_summary(json_file)
        if summary:
            summaries.append(summary)
    
    if not summaries:
        print("❌ No valid benchmark files found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(summaries)
    
    # Sort by model, type, timestamp
    df = df.sort_values(['Model', 'Type', 'Timestamp'])
    
    print(f"\n📊 BENCHMARK SUMMARY TABLE")
    print("=" * 120)
    print(f"Total benchmark files processed: {len(df)}")
    print(f"Unique models: {', '.join(df['Model'].unique())}")
    print()
    
    # Display the full table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df.to_string(index=False))
    
    # Create aggregated summary by model
    print(f"\n📈 AGGREGATED SUMMARY BY MODEL")
    print("=" * 80)
    
    model_summary = df.groupby(['Model', 'Type']).agg({
        'Num_Calls': 'sum',
        'Total_Prompt_Tokens': 'sum',
        'Total_Completion_Tokens': 'sum',
        'Total_Tokens': 'sum',
        'Avg_Response_Time_s': 'mean',
        'Avg_TPS': 'mean',
        'Total_Cost_USD': 'sum',
        'Cache_Hit_Rate_%': 'mean',
    }).round({
        'Avg_Response_Time_s': 3,
        'Avg_TPS': 2,
        'Total_Cost_USD': 6,
        'Cache_Hit_Rate_%': 2
    })
    
    print(model_summary.to_string())
    
    # Overall totals
    print(f"\n🎯 OVERALL TOTALS ACROSS ALL RUNS")
    print("=" * 50)
    print(f"Total API Calls: {df['Num_Calls'].sum():,}")
    print(f"Total Prompt Tokens: {df['Total_Prompt_Tokens'].sum():,}")
    print(f"Total Completion Tokens: {df['Total_Completion_Tokens'].sum():,}")
    print(f"Total Tokens: {df['Total_Tokens'].sum():,}")
    print(f"Total Cost: ${df['Total_Cost_USD'].sum():.6f}")
    print(f"Average TPS (across all runs): {df['Avg_TPS'].mean():.2f}")
    print(f"Average Response Time (across all runs): {df['Avg_Response_Time_s'].mean():.3f}s")
    
    # Cache performance (API only)
    api_runs = df[df['Type'] == 'API']
    if not api_runs.empty:
        total_cached = api_runs['Cached_Tokens'].sum()
        total_prompt_api = api_runs['Total_Prompt_Tokens'].sum()
        overall_cache_rate = (total_cached / total_prompt_api * 100) if total_prompt_api > 0 else 0
        print(f"Overall Cache Hit Rate (API runs): {overall_cache_rate:.2f}%")
    
    # Save to file if specified
    if output_file:
        try:
            if output_file.endswith('.csv'):
                df.to_csv(output_file, index=False)
                print(f"\n💾 Summary table saved to: {output_file}")
            elif output_file.endswith('.xlsx'):
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Benchmark_Summary', index=False)
                    model_summary.to_excel(writer, sheet_name='Model_Aggregated')
                print(f"\n💾 Summary table saved to: {output_file}")
            else:
                # Save as tab-separated for easy copying
                df.to_csv(output_file, index=False, sep='\t')
                print(f"\n💾 Summary table saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving file: {e}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive summary table of all benchmark results")
    parser.add_argument("--results-dir", type=str, default="results", 
                       help="Directory containing benchmark JSON files (default: results)")
    parser.add_argument("--output", type=str, 
                       help="Output file to save the summary table (CSV, XLSX, or TSV)")
    args = parser.parse_args()
    
    df = create_benchmark_summary_table(args.results_dir, args.output)
    
    if df is not None:
        print(f"\n✅ Summary table generation complete!")
        if not args.output:
            print("💡 Use --output filename.csv to save the table to a file")

if __name__ == "__main__":
    main() 