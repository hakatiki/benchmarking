import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

def load_all_benchmarks():
    results_path = Path("results")
    summaries = []
    
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract model name from JSON metadata first, then fallback to filename parsing
            model = data.get("benchmark_info", {}).get("model", "")
            
            # If not in metadata, extract from filename and normalize
            if not model:
                filename = json_file.name
                if "gpt_4o_mini" in filename:
                    model = "gpt-4o mini"  # Match pricing.json key (space, not dash)
                elif "gpt_4o" in filename:
                    model = "gpt-4o"  
                elif "gpt_4_1" in filename:
                    model = "gpt-4.1"
                else:
                    model = filename.split('_')[2]
            
            # Normalize model names to match pricing.json format
            if model == "gpt-4o-mini":
                model = "gpt-4o mini"  # Convert dash to space
            
            results = data.get("results", [])
            successful_calls = [r for r in results if 'error' not in r]
            
            if successful_calls:
                tps_values = [r.get('tokens_per_second', 0) for r in successful_calls]
                
                # Calculate confidence interval for TPS
                n = len(tps_values)
                mean_tps = np.mean(tps_values)
                std_tps = np.std(tps_values, ddof=1)  # Sample standard deviation
                se_tps = std_tps / np.sqrt(n)  # Standard error
                
                # 95% confidence interval using t-distribution
                confidence_level = 0.95
                alpha = 1 - confidence_level
                t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
                margin_error = t_critical * se_tps
                ci_lower = mean_tps - margin_error
                ci_upper = mean_tps + margin_error
                
                summaries.append({
                    'Model': model,  # Use the normalized model name
                    'Calls': len(successful_calls),
                    'Total_Prompt_Tokens': sum(r.get('prompt_tokens', 0) for r in successful_calls),
                    'Total_Completion_Tokens': sum(r.get('completion_tokens', 0) for r in successful_calls),
                    'Total_Tokens': sum(r.get('total_tokens', 0) for r in successful_calls),
                    'Avg_TPS': round(mean_tps, 2),
                    'TPS_CI_Lower': round(ci_lower, 2),
                    'TPS_CI_Upper': round(ci_upper, 2),
                    'TPS_StdDev': round(std_tps, 2),
                    'Avg_Response_Time': round(np.mean([r.get('response_time_seconds', 0) for r in successful_calls]), 3),
                    'Total_Cost': round(sum(r.get('cost_usd', {}).get('total_cost', 0) for r in successful_calls), 6),
                    'Cached_Tokens': sum(r.get('cached_tokens', 0) for r in successful_calls),
                })
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    return summaries

def main():
    summaries = load_all_benchmarks()
    
    print("📊 BENCHMARK SUMMARY TABLE WITH CONFIDENCE INTERVALS")
    print("=" * 100)
    print(f"{'Model':<12} {'Calls':<6} {'Input':<8} {'Output':<8} {'Avg TPS':<8} {'95% CI TPS':<15} {'±StdDev':<8} {'Avg RT':<8} {'Cost':<10}")
    print("-" * 100)
    
    total_calls = 0
    total_input = 0
    total_output = 0
    total_cost = 0
    total_cached = 0
    all_tps_values = []
    
    for s in summaries:
        ci_range = f"[{s['TPS_CI_Lower']}-{s['TPS_CI_Upper']}]"
        std_dev = f"±{s['TPS_StdDev']}"
        
        print(f"{s['Model']:<12} {s['Calls']:<6} {s['Total_Prompt_Tokens']:<8} {s['Total_Completion_Tokens']:<8} {s['Avg_TPS']:<8} {ci_range:<15} {std_dev:<8} {s['Avg_Response_Time']:<8} ${s['Total_Cost']:<9}")
        
        total_calls += s['Calls']
        total_input += s['Total_Prompt_Tokens']
        total_output += s['Total_Completion_Tokens']
        total_cost += s['Total_Cost']
        total_cached += s['Cached_Tokens']
        all_tps_values.append(s['Avg_TPS'])
    
    print("-" * 100)
    
    # Overall statistics
    overall_avg_tps = round(np.mean(all_tps_values), 2)
    overall_tps_std = round(np.std(all_tps_values, ddof=1), 2) if len(all_tps_values) > 1 else 0
    
    # Overall confidence interval (treating model averages as samples)
    if len(all_tps_values) > 1:
        n_models = len(all_tps_values)
        se_overall = overall_tps_std / np.sqrt(n_models)
        t_crit_overall = stats.t.ppf(0.975, df=n_models-1)
        margin_overall = t_crit_overall * se_overall
        ci_lower_overall = overall_avg_tps - margin_overall
        ci_upper_overall = overall_avg_tps + margin_overall
        overall_ci = f"[{ci_lower_overall:.2f}-{ci_upper_overall:.2f}]"
    else:
        overall_ci = "N/A"
    
    print(f"{'OVERALL':<12} {total_calls:<6} {total_input:<8} {total_output:<8} {overall_avg_tps:<8} {overall_ci:<15} ±{overall_tps_std:<7} {'N/A':<8} ${total_cost:<9}")
    
    print()
    print("📈 PERFORMANCE STATISTICS:")
    print(f"Total API Calls: {total_calls:,}")
    print(f"Total Tokens Processed: {total_input + total_output:,}")
    print(f"Overall Cache Hit Rate: {(total_cached / total_input * 100):.2f}%" if total_input > 0 else "Cache Hit Rate: 0%")
    print(f"Total Cost: ${total_cost:.6f}")
    print()
    print("🔍 TPS ANALYSIS:")
    print(f"Mean TPS across models: {overall_avg_tps} tokens/second")
    print(f"95% Confidence Interval: {overall_ci}")
    print(f"Standard deviation: ±{overall_tps_std}")
    print()
    print("💡 Note: 95% CI shows the range where the true mean likely falls.")
    print("   Smaller intervals indicate more consistent performance.")

if __name__ == "__main__":
    main() 