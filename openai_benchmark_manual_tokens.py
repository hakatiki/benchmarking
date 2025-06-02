import time
import os
from openai import OpenAI
from typing import List, Dict
import json
import argparse
from datetime import datetime
import tiktoken


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class OpenAIBenchmark:
    def __init__(self, api_key: str = None, prompts_file: str = "prompts.json", pricing_file: str = "pricing.json", model: str = "gpt-4o"):
        """Initialize the OpenAI client and load prompts from file."""
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.results = []
        self.prompts = self.load_prompts(prompts_file)
        self.pricing = self.load_pricing(pricing_file)
        self.default_model = model
        self.start_time = datetime.now()
        
        # Initialize tokenizer for manual token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for newer models
            print(f"⚠️  Tokenizer not found for {model}, using cl100k_base fallback")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def load_prompts(self, filename: str) -> dict:
        """Load prompts from JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            print(f"✅ Loaded prompts from {filename}")
            return prompts
        except FileNotFoundError:
            print(f"❌ Prompts file {filename} not found. Using default prompts.")
            return {
                "basic_prompts": [
                    "Hello, how are you?",
                    "Explain quantum computing in simple terms.",
                    "Write a short poem about artificial intelligence."
                ],
                "caching_tests": [
                    {"name": "Simple test", "prompt": "Hello, how are you today?", "repetitions": 3}
                ],
                "rapid_fire_tests": [
                    {"name": "Quick test", "prompt": "What is 2+2?", "repetitions": 3}
                ]
            }
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing {filename}: {e}")
            print("Using default prompts.")
            return {"basic_prompts": ["Hello, how are you?"], "caching_tests": [], "rapid_fire_tests": []}
    
    def load_pricing(self, filename: str) -> dict:
        """Load pricing data from JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                pricing = json.load(f)
            print(f"✅ Loaded pricing data from {filename}")
            return pricing
        except FileNotFoundError:
            print(f"❌ Pricing file {filename} not found. Using fallback pricing.")
            return {
                "openai": {
                    "gpt-4o": {
                        "input_usd_per_1m": 5.00,
                        "output_usd_per_1m": 20.00,
                        "cached_input_usd_per_1m": 2.50
                    }
                }
            }
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing {filename}: {e}")
            print("Using fallback pricing.")
            return {"openai": {"gpt-4o": {"input_usd_per_1m": 5.00, "output_usd_per_1m": 20.00}}}
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def llm_wrapper(self, prompt: str, model: str) -> str:
        """Wrapper that only returns the completion string, nothing else."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict:
        """Calculate accurate cost based on model and token usage."""
        # Extract provider and model name
        provider = "openai"  # Default for now, could be extracted from model name
        model_key = model
        
        # Get pricing for the model
        if provider in self.pricing and model_key in self.pricing[provider]:
            prices = self.pricing[provider][model_key]
        else:
            # Fallback to gpt-4o pricing if model not found
            print(f"⚠️  Pricing not found for {model}, using gpt-4o fallback")
            prices = self.pricing.get("openai", {}).get("gpt-4o", {
                "input_usd_per_1m": 5.00,
                "output_usd_per_1m": 20.00,
                "cached_input_usd_per_1m": 2.50
            })
        
        # Calculate costs
        uncached_tokens = prompt_tokens - cached_tokens
        
        input_cost = (uncached_tokens * prices.get("input_usd_per_1m", 5.0)) / 1_000_000
        cached_cost = (cached_tokens * prices.get("cached_input_usd_per_1m", 2.5)) / 1_000_000
        output_cost = (completion_tokens * prices.get("output_usd_per_1m", 20.0)) / 1_000_000
        
        total_cost = input_cost + cached_cost + output_cost
        
        # Calculate what would have been paid without caching
        would_have_paid_input = (prompt_tokens * prices.get("input_usd_per_1m", 5.0)) / 1_000_000
        total_without_cache = would_have_paid_input + output_cost
        savings = total_without_cache - total_cost
        
        return {
            "input_cost_usd": input_cost,
            "cached_cost_usd": cached_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
            "savings_from_cache_usd": savings,
            "total_without_cache_usd": total_without_cache
        }
    
    def benchmark_single_call(self, prompt: str, model: str = "gpt-4o") -> Dict:
        """Make a single API call and benchmark it with manual token counting."""
        print(f"Testing prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        # Count prompt tokens manually
        prompt_tokens = self.count_tokens(prompt)
        
        start_time = time.time()
        
        try:
            # Use wrapper to get only completion text
            response_content = self.llm_wrapper(prompt, model)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Count completion tokens manually
            completion_tokens = self.count_tokens(response_content)
            total_tokens = prompt_tokens + completion_tokens
            
            # Cached tokens are always 0 since we can't detect them without API response
            cached_tokens = 0
            
            # Calculate costs
            cost_info = self.calculate_cost(model, prompt_tokens, completion_tokens, cached_tokens)
            
            result = {
                "prompt": prompt,
                "response": response_content,
                "response_time_seconds": round(response_time, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,  # Always 0 with manual counting
                "uncached_prompt_tokens": prompt_tokens,  # Always equals prompt_tokens
                "tokens_per_second": round(completion_tokens / response_time, 2) if response_time > 0 else 0,
                "model": model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "token_counting_method": "manual_tiktoken",  # Mark that we used manual counting
                # Add cost information
                "cost_usd": {
                    "input_cost": round(cost_info["input_cost_usd"], 6),
                    "cached_cost": round(cost_info["cached_cost_usd"], 6),  # Always 0
                    "output_cost": round(cost_info["output_cost_usd"], 6),
                    "total_cost": round(cost_info["total_cost_usd"], 6),
                    "savings_from_cache": round(cost_info["savings_from_cache_usd"], 6),  # Always 0
                    "total_without_cache": round(cost_info["total_without_cache_usd"], 6)
                }
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            return {"error": str(e), "prompt": prompt}
    
    def run_caching_test(self, prompt: str, repetitions: int = 3, model: str = None):
        """Test caching by sending the same prompt multiple times rapidly."""
        if model is None:
            model = self.default_model
        print(f"\n🎯 CACHING TEST: Sending same prompt {repetitions} times rapidly")
        print(f"Model: {model}")
        print(f"⚠️  Note: Cached tokens always 0 with manual counting")
        print(f"Prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        print("-" * 50)
        
        for i in range(repetitions):
            print(f"\n🔄 Attempt {i+1}/{repetitions}")
            result = self.benchmark_single_call(prompt, model)
            
            if "error" not in result:
                print(f"✅ Response time: {result['response_time_seconds']}s")
                print(f"📝 Tokens - Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}")
                print(f"🔢 Manual token counting (no cache detection)")
                print(f"⚡ Tokens per second: {result['tokens_per_second']}")
                # Show cost breakdown
                cost = result['cost_usd']
                print(f"💵 Cost: ${cost['total_cost']:.6f} (Input: ${cost['input_cost']:.6f} + Output: ${cost['output_cost']:.6f})")
            else:
                print(f"❌ Failed: {result['error']}")
            
            # Very short delay to be respectful but still test rapid requests
            if i < repetitions - 1:  # Don't delay after last request
                time.sleep(0.2)
    
    def run_benchmark_suite(self):
        """Run a suite of benchmark tests with different prompt types."""
        print("🚀 Starting OpenAI API Benchmark Suite (Manual Token Counting)")
        print("=" * 50)
        print(f"🤖 Model: {self.default_model}")
        print(f"🔢 Token counting: Manual (tiktoken)")
        print(f"⚠️  Cached tokens always 0 (cannot detect without API response)")
        print(f"⏰ Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        # First run basic prompts
        basic_prompts = self.prompts.get("basic_prompts", [])
        if basic_prompts:
            print(f"Running {len(basic_prompts)} basic prompts...")
            for i, prompt in enumerate(basic_prompts, 1):
                print(f"\n📊 Basic Test {i}/{len(basic_prompts)}")
                result = self.benchmark_single_call(prompt, self.default_model)
                
                if "error" not in result:
                    print(f"✅ Response time: {result['response_time_seconds']}s")
                    print(f"📝 Tokens - Prompt: {result['prompt_tokens']}, Completion: {result['completion_tokens']}, Total: {result['total_tokens']}")
                    print(f"🔢 Manual counting (cached always 0)")
                    print(f"⚡ Tokens per second: {result['tokens_per_second']}")
                    # Show cost
                    cost = result['cost_usd']
                    print(f"💵 Cost: ${cost['total_cost']:.6f}")
                    print(f"💬 Response preview: {result['response'][:100]}{'...' if len(result['response']) > 100 else ''}")
                else:
                    print(f"❌ Failed: {result['error']}")
                
                # Short delay between different prompts
                time.sleep(0.5)
        
        # Now test caching with repeated requests
        caching_tests = self.prompts.get("caching_tests", [])
        if caching_tests:
            print("\n" + "=" * 50)
            print("🎯 CACHING PERFORMANCE TESTS")
            print("=" * 50)
            
            for test in caching_tests:
                print(f"\n🎯 CACHING TEST: {test['name']}")
                self.run_caching_test(test['prompt'], test['repetitions'], self.default_model)
        
        # Rapid fire tests
        rapid_fire_tests = self.prompts.get("rapid_fire_tests", [])
        if rapid_fire_tests:
            print("\n" + "=" * 50)
            print("🚀 RAPID FIRE TESTS")
            print("=" * 50)
            
            for test in rapid_fire_tests:
                print(f"\n🚀 RAPID FIRE TEST: {test['name']} - {test['repetitions']} times with minimal delay")
                for i in range(test['repetitions']):
                    print(f"\n⚡ Rapid fire {i+1}/{test['repetitions']}")
                    result = self.benchmark_single_call(test['prompt'], self.default_model)
                    if "error" not in result:
                        cost = result['cost_usd']
                        print(f"Time: {result['response_time_seconds']}s | Manual tokens | Speed: {result['tokens_per_second']} tok/s | Cost: ${cost['total_cost']:.6f}")
                    # Minimal delay for rapid fire
                    time.sleep(0.1)
    
    def print_summary(self):
        """Print a summary of all benchmark results."""
        if not self.results:
            print("No successful results to summarize.")
            return
        
        successful_results = [r for r in self.results if "error" not in r]
        
        if not successful_results:
            print("No successful API calls to analyze.")
            return
        
        print("\n" + "=" * 50)
        print("📈 BENCHMARK SUMMARY (Manual Token Counting)")
        print("=" * 50)
        
        total_calls = len(successful_results)
        avg_response_time = sum(r['response_time_seconds'] for r in successful_results) / total_calls
        avg_prompt_tokens = sum(r['prompt_tokens'] for r in successful_results) / total_calls
        avg_completion_tokens = sum(r['completion_tokens'] for r in successful_results) / total_calls
        avg_total_tokens = sum(r['total_tokens'] for r in successful_results) / total_calls
        avg_tokens_per_second = sum(r['tokens_per_second'] for r in successful_results) / total_calls
        
        min_response_time = min(r['response_time_seconds'] for r in successful_results)
        max_response_time = max(r['response_time_seconds'] for r in successful_results)
        
        total_tokens_used = sum(r['total_tokens'] for r in successful_results)
        total_cached_tokens = 0  # Always 0 with manual counting
        total_uncached_tokens = sum(r['uncached_prompt_tokens'] for r in successful_results)
        
        print(f"🔢 Total API calls: {total_calls}")
        print(f"⏱️  Average response time: {avg_response_time:.3f}s")
        print(f"⚡ Fastest response: {min_response_time:.3f}s")
        print(f"🐌 Slowest response: {max_response_time:.3f}s")
        print(f"📝 Average prompt tokens: {avg_prompt_tokens:.1f}")
        print(f"💬 Average completion tokens: {avg_completion_tokens:.1f}")
        print(f"📊 Average total tokens: {avg_total_tokens:.1f}")
        print(f"🚀 Average tokens per second: {avg_tokens_per_second:.2f}")
        print(f"💰 Total tokens consumed: {total_tokens_used}")
        
        # Manual counting notes
        print(f"🔢 Token counting method: Manual (tiktoken)")
        print(f"⚠️  Cached tokens: Always 0 (cannot detect without API response)")
        print(f"📊 All prompt tokens counted as uncached: {total_uncached_tokens}")
        
        # Cost estimation using accurate pricing from individual results
        total_input_cost = sum(r['cost_usd']['input_cost'] for r in successful_results)
        total_cached_cost = 0  # Always 0
        total_output_cost = sum(r['cost_usd']['output_cost'] for r in successful_results)
        total_cost = sum(r['cost_usd']['total_cost'] for r in successful_results)
        total_savings = 0  # Always 0 with manual counting
        total_without_cache = sum(r['cost_usd']['total_without_cache'] for r in successful_results)
        
        print(f"💵 Total cost: ${total_cost:.6f}")
        print(f"   • Input cost: ${total_input_cost:.6f}")
        print(f"   • Cached cost: ${total_cached_cost:.6f} (always 0)")
        print(f"   • Output cost: ${total_output_cost:.6f}")
        
        print(f"⚠️  No cache savings detected (manual counting limitation)")
    
    def save_results(self, filename: str = None):
        """Save results to a JSON file with model and timestamp in filename."""
        if filename is None:
            # Generate filename with model and timestamp
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            model_safe = self.default_model.replace("-", "_").replace(".", "_")
            filename = f"results/manual_{model_safe}_{timestamp}.json"
        
        # Add metadata to results
        metadata = {
            "benchmark_info": {
                "model": self.default_model,
                "token_counting_method": "manual_tiktoken",
                "cached_token_detection": False,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "total_requests": len(self.results),
                "successful_requests": len([r for r in self.results if "error" not in r])
            },
            "results": self.results
        }
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {filename}")
        return filename

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark OpenAI API performance with manual token counting')
    parser.add_argument('--model', '-m', type=str, default='gpt-4o',
                       help='OpenAI model to benchmark (default: gpt-4o)')
    parser.add_argument('--prompts', '-p', type=str, default='prompts.json',
                       help='Path to prompts JSON file (default: prompts.json)')
    parser.add_argument('--pricing', type=str, default='pricing.json',
                       help='Path to pricing JSON file (default: pricing.json)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output filename (default: auto-generated with model and timestamp)')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models from pricing file and exit')
    
    args = parser.parse_args()
    
    print("🤖 OpenAI API Benchmark Tool (Manual Token Counting)")
    print("🔢 This version counts tokens manually using tiktoken")
    print("⚠️  Cached tokens always 0 (cannot detect without API response)")
    print("Make sure you have set your OPENAI_API_KEY environment variable!")
    print("Tip: Create a .env file with OPENAI_API_KEY=your_key_here")
    print()
    
    # Initialize the benchmark
    benchmark = OpenAIBenchmark(
        prompts_file=args.prompts, 
        pricing_file=args.pricing,
        model=args.model
    )
    
    # List models if requested
    if args.list_models:
        print("📋 Available models in pricing file:")
        for provider, models in benchmark.pricing.items():
            print(f"\n{provider.upper()}:")
            for model_name in models.keys():
                print(f"  • {model_name}")
        return
    
    print(f"🎯 Selected model: {args.model}")
    print()
    
    # Run the benchmark suite
    benchmark.run_benchmark_suite()
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results(args.output)

if __name__ == "__main__":
    main() 