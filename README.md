📊 BENCHMARK SUMMARY TABLE WITH CONFIDENCE INTERVALS
====================================================================================================
Model        Calls  Input    Output   Avg TPS  95% CI TPS      ±StdDev  Avg RT   Cost
----------------------------------------------------------------------------------------------------
gpt-4o       65     5773     32263    55.38    [51.68-59.08]   ±14.94   9.471    $0.670285
gpt-4o mini  65     5773     35638    51.33    [48.52-54.15]   ±11.35   10.711   $0.088075
gpt-4.1      65     5773     30508    58.05    [53.81-62.3]    ±17.12   8.861    $0.251002
----------------------------------------------------------------------------------------------------
OVERALL      195    17319    98409    54.92    [46.52-63.32]   ±3.38    N/A      $1.009362

📈 PERFORMANCE STATISTICS:
Total API Calls: 195
Total Tokens Processed: 115,728
Overall Cache Hit Rate: 44.34%
Total Cost: $1.009362

🔍 TPS ANALYSIS:
Mean TPS across models: 54.92 tokens/second
95% Confidence Interval: [46.52-63.32]
Standard deviation: ±3.38

💡 Note: 95% CI shows the range where the true mean likely falls.
   Smaller intervals indicate more consistent performance.
PS C:\Users\takat\OneDrive\Documents\Programming\benchmarking> 


# OpenAI API Benchmark Tool

A simple Python script to benchmark the OpenAI GPT-4o API, measuring response times, token usage, and cached token performance. All test prompts are configurable via a JSON file, with command line support for easy model switching.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key using one of these methods:

   **Option A: Environment variable**
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # Linux/Mac
   export OPENAI_API_KEY=your_api_key_here
   ```

   **Option B: .env file (recommended)**
   ```bash
   # Create a .env file in the project directory
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## Usage

### Basic Usage
Run the benchmark with default settings (GPT-4o):
```bash
python openai_benchmark.py
```

### Command Line Options
```bash
# Use a different model
python openai_benchmark.py --model gpt-4o-mini

# Short form
python openai_benchmark.py -m gpt-4o-mini

# Use custom prompts file
python openai_benchmark.py --prompts my_prompts.json

# Custom output filename
python openai_benchmark.py --output my_results.json

# List available models
python openai_benchmark.py --list-models

# Combine options
python openai_benchmark.py -m gpt-4o-mini -p custom_prompts.json -o mini_test.json
```

### Available Command Line Arguments
- `--model, -m`: OpenAI model to benchmark (default: gpt-4o)
- `--prompts, -p`: Path to prompts JSON file (default: prompts.json)  
- `--pricing`: Path to pricing JSON file (default: pricing.json)
- `--output, -o`: Output filename (default: auto-generated with model and timestamp)
- `--list-models`: List available models from pricing file and exit

## Output Files

Results are automatically saved with descriptive filenames:
- Format: `benchmark_results_{model}_{timestamp}.json`
- Example: `benchmark_results_gpt_4o_mini_20241215_143022.json`

Each results file includes:
- **Benchmark metadata**: Model, timestamps, duration, request counts
- **Individual results**: All API call data with costs
- **UTF-8 encoding**: Supports international characters

## Customizing Test Prompts

The script loads test prompts from `prompts.json`. You can customize this file to test different scenarios:

### File Structure
```json
{
  "basic_prompts": [
    "Hello, how are you?",
    "Your basic test prompts here..."
  ],
  "caching_tests": [
    {
      "name": "Test name",
      "prompt": "Your prompt to test caching",
      "repetitions": 3
    }
  ],
  "rapid_fire_tests": [
    {
      "name": "Rapid test name", 
      "prompt": "Quick prompt for rapid testing",
      "repetitions": 5
    }
  ]
}
```

### Test Types
- **basic_prompts**: Run once each to establish baseline performance
- **caching_tests**: Run multiple times rapidly to test prompt caching
- **rapid_fire_tests**: Run with minimal delays to stress test caching

## What it measures

- **Response time**: How long each API call takes
- **Token usage**: Prompt tokens, completion tokens, and total tokens
- **Cached tokens**: Tokens that were cached from previous requests (cost savings!)
- **Tokens per second**: Generation speed
- **Cache hit rate**: Percentage of prompt tokens that were cached
- **Cost estimation**: Accurate cost calculation including caching savings

## Features

- **Command line model selection** - Easy switching between models
- **Auto-generated filenames** - Include model and timestamp
- **Configurable prompts** - Edit `prompts.json` to customize all tests
- **Accurate pricing** - Model-specific pricing from `pricing.json`
- **Cached token tracking** - Shows when OpenAI's prompt caching saves you money
- **Real-time USD costs** - See exact costs for each API call
- **Comprehensive metadata** - Detailed benchmark information in results
- **Data export** - Saves detailed results to JSON file
- **Error handling** - Graceful handling of API errors and missing files

## Output

The script will:
1. Load prompts from `prompts.json` (or use defaults if file not found)
2. Load pricing from `pricing.json` for accurate cost calculations
3. Run basic prompts once each for baseline measurements
4. Run caching tests with repeated prompts to demonstrate caching
5. Run rapid-fire tests with minimal delays
6. Display real-time results including cached token info and USD costs
7. Show a comprehensive summary with cache statistics
8. Save detailed results to timestamped JSON file

## Sample Output

```
✅ Loaded prompts from prompts.json
✅ Loaded pricing data from pricing.json
🚀 Starting OpenAI API Benchmark Suite
==================================================
🤖 Model: gpt-4o-mini
⏰ Started: 2024-12-15 14:30:22
==================================================
Running 5 basic prompts...

📊 Basic Test 1/5
Testing prompt: 'Hello, how are you?'
✅ Response time: 1.234s
📝 Tokens - Prompt: 15, Completion: 25, Total: 40
💭 No cached tokens (first time or caching not available)
⚡ Tokens per second: 20.26
💵 Cost: $0.000069

🎯 CACHING PERFORMANCE TESTS
==================================================

🎯 CACHING TEST: Long Hungarian recycling text
Model: gpt-4o-mini

🔄 Attempt 2/3
🎯 Cached tokens: 412 (💰 CACHE HIT!)
📈 Cache percentage: 100.0%
💵 Cost: $0.001456 (Input: $0.000031 + Cached: $0.000124 + Output: $0.001301)
💰 Cache savings: $0.000124

💾 Results saved to benchmark_results_gpt_4o_mini_20241215_143022.json
```

## Understanding Cached Tokens

OpenAI's prompt caching can significantly reduce costs for repeated or similar prompts:

- **Cached tokens** are charged at 50% of the normal rate
- **Cache hit rate** shows what percentage of your prompt tokens were cached
- The benchmark shows both actual cost and what you would have paid without caching
- Configure your own prompts in `prompts.json` to test specific caching scenarios

## Examples

```bash
# Compare different models
python openai_benchmark.py -m gpt-4o
python openai_benchmark.py -m gpt-4o-mini  
python openai_benchmark.py -m o3

# Test with custom prompts
python openai_benchmark.py -m gpt-4o-mini -p my_test_prompts.json

# See what models are available
python openai_benchmark.py --list-models
``` 