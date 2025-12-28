"""
ONNX Runtime Web Benchmark Automation Script (CLI Version)

This script automates running benchmarks for each model individually,
clearing cache between runs, and collecting results into a CSV file.

Requirements:
    pip install playwright pandas
    playwright install chromium

Usage:
    python run_benchmark_cli.py --url http://localhost:5173 --models ./public/models/models.json
    
    python run_benchmark_cli.py --help  # for all options
"""

import asyncio
import json
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright
import pandas as pd


class BenchmarkRunner:
    def __init__(self, url, models_config_path, output_dir="benchmark_results", headless=False):
        self.url = url
        self.models_config_path = models_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.headless = headless
        
    async def load_models_list(self):
        """Load the list of models from models.json"""
        with open(self.models_config_path, 'r') as f:
            config = json.load(f)
        return [model['name'] for model in config.get('models', [])]
    
    async def run_single_model(self, browser, model_name, backend="wasm", warmup=1, measure=100):
        """Run benchmark for a single model"""
        print(f"\n{'='*80}")
        print(f"Running benchmark: {model_name} (backend: {backend})")
        print(f"{'='*80}")
        
        # Create new browser context with cache cleared
        context = await browser.new_context()
        await context.clear_cookies()
        
        page = await context.new_page()
        
        # Enable console logging
        logs = []
        page.on("console", lambda msg: logs.append(msg.text()))
        
        try:
            # Navigate to the benchmark page
            print(f"Loading page: {self.url}")
            await page.goto(self.url, wait_until="networkidle", timeout=60000)
            
            # Wait for the page to be ready
            await page.wait_for_selector("#runBtn", timeout=30000)
            print("Page loaded successfully")
            
            # Select backend
            await page.select_option("#backendSelect", backend)
            print(f"Selected backend: {backend}")
            
            # Set warmup and measure runs
            await page.fill("#warmupRuns", str(warmup))
            await page.fill("#measureRuns", str(measure))
            print(f"Set warmup={warmup}, measure={measure}")
            
            # Wait a bit for the models to load
            await asyncio.sleep(2)
            
            # Select only the target model
            # First, get all options
            options = await page.locator("#modelsSelect option").all()
            
            # Find and select only our target model
            found = False
            for option in options:
                value = await option.get_attribute("value")
                if value == model_name:
                    # Use JavaScript to select this specific option
                    await page.evaluate(f"""
                        const select = document.getElementById('modelsSelect');
                        for (let opt of select.options) {{
                            opt.selected = (opt.value === '{model_name}');
                        }}
                    """)
                    found = True
                    print(f"Selected model: {model_name}")
                    break
            
            if not found:
                raise Exception(f"Model '{model_name}' not found in the models list")
            
            # Click Run button
            print("Starting benchmark...")
            await page.click("#runBtn")
            
            # Wait for benchmark to complete (check if Run button is re-enabled)
            # This might take a long time depending on the model
            timeout_ms = 600000  # 10 minutes timeout
            start_time = time.time()
            last_log_time = start_time
            
            while True:
                # Check if run button is enabled (benchmark complete)
                is_disabled = await page.locator("#runBtn").is_disabled()
                
                if not is_disabled:
                    print("Benchmark completed!")
                    break
                
                # Check timeout
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > timeout_ms:
                    raise Exception(f"Benchmark timed out after {timeout_ms/1000} seconds")
                
                # Log progress every 5 seconds
                current_time = time.time()
                if current_time - last_log_time >= 5:
                    try:
                        status_text = await page.locator("#status").text_content()
                        last_lines = status_text.strip().split('\n')[-3:]
                        print(f"Progress: {' | '.join(last_lines)}")
                    except:
                        pass
                    last_log_time = current_time
                
                await asyncio.sleep(1)
            
            # Extract results from the page
            await asyncio.sleep(2)  # Wait a bit for results to be written
            
            status_text = await page.locator("#status").text_content()
            
            # Parse results from status text
            result = self.parse_results(model_name, backend, status_text)
            
            # Also try to get from table if available
            table_rows = await page.locator("#resultsBody tr").all()
            if table_rows:
                for row in table_rows:
                    cells = await row.locator("td").all()
                    if cells and len(cells) >= 7:
                        table_model = await cells[0].text_content()
                        if table_model == model_name:
                            result['warmup_runs'] = await cells[2].text_content()
                            result['avg_warmup_ms'] = await cells[3].text_content()
                            result['measure_runs'] = await cells[4].text_content()
                            result['avg_inference_ms'] = await cells[5].text_content()
                            result['notes'] = await cells[6].text_content()
                            break
            
            print(f"\nResults for {model_name}:")
            print(f"  Avg Warmup: {result.get('avg_warmup_ms', 'N/A')} ms")
            print(f"  Avg Inference: {result.get('avg_inference_ms', 'N/A')} ms")
            print(f"  Status: {result.get('status', 'N/A')}")
            
            self.results.append(result)
            
            return result
            
        except Exception as e:
            print(f"ERROR: Failed to run benchmark for {model_name}: {str(e)}")
            error_result = {
                'timestamp': datetime.now().isoformat(),
                'model': model_name,
                'backend': backend,
                'warmup_runs': warmup,
                'measure_runs': measure,
                'avg_warmup_ms': 'ERROR',
                'avg_inference_ms': 'ERROR',
                'status': 'FAILED',
                'notes': str(e),
                'error': str(e)
            }
            self.results.append(error_result)
            return error_result
            
        finally:
            # Close context and clear cache
            await context.close()
            print(f"Closed browser context for {model_name}")
    
    def parse_results(self, model_name, backend, status_text):
        """Parse results from the status text"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'backend': backend,
            'warmup_runs': 'N/A',
            'measure_runs': 'N/A',
            'avg_warmup_ms': 'N/A',
            'avg_inference_ms': 'N/A',
            'status': 'UNKNOWN',
            'notes': '',
            'session_create_ms': 'N/A'
        }
        
        # Try to extract success message
        success_pattern = r'SUCCESS: warmup=([\d.]+) ms, inference=([\d.]+) ms'
        success_match = re.search(success_pattern, status_text)
        
        if success_match:
            result['avg_warmup_ms'] = success_match.group(1)
            result['avg_inference_ms'] = success_match.group(2)
            result['status'] = 'SUCCESS'
        
        # Try to extract DONE message (older format)
        done_pattern = r'DONE: warmup=([\d.]+) ms, inference=([\d.]+) ms'
        done_match = re.search(done_pattern, status_text)
        
        if done_match:
            result['avg_warmup_ms'] = done_match.group(1)
            result['avg_inference_ms'] = done_match.group(2)
            result['status'] = 'SUCCESS'
        
        # Check for failure
        if 'FAILED:' in status_text or 'ERROR' in status_text:
            result['status'] = 'FAILED'
            # Try to extract error message
            failed_match = re.search(r'FAILED: (.+?)(?:\n|$)', status_text)
            if failed_match:
                result['notes'] = failed_match.group(1)
        
        # Extract session creation time
        session_match = re.search(r'session_create_ms=([\d.]+)', status_text)
        if session_match:
            result['session_create_ms'] = session_match.group(1)
        
        return result
    
    async def run_all_models(self, backend="wasm", warmup=1, measure=100, models_filter=None):
        """Run benchmarks for all models"""
        all_models = await self.load_models_list()
        
        # Filter models if needed
        if models_filter:
            models = [m for m in all_models if any(f.lower() in m.lower() for f in models_filter)]
            print(f"\nFiltered to {len(models)} models matching: {models_filter}")
        else:
            models = all_models
        
        if not models:
            print("No models to benchmark!")
            return
        
        print(f"\n{'='*80}")
        print(f"Starting benchmark run for {len(models)} models")
        print(f"Backend: {backend}, Warmup: {warmup}, Measure: {measure}")
        print(f"Headless: {self.headless}")
        print(f"{'='*80}\n")
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                headless=self.headless,
                args=['--disable-cache', '--disk-cache-size=0']
            )
            
            try:
                for i, model_name in enumerate(models, 1):
                    print(f"\n[{i}/{len(models)}] Processing model: {model_name}")
                    
                    await self.run_single_model(
                        browser, 
                        model_name, 
                        backend=backend,
                        warmup=warmup,
                        measure=measure
                    )
                    
                    # Small delay between models
                    await asyncio.sleep(2)
                
            finally:
                await browser.close()
                print("\nBrowser closed")
        
        print(f"\n{'='*80}")
        print(f"All benchmarks completed! Total models: {len(models)}")
        print(f"Successful: {sum(1 for r in self.results if r.get('status') == 'SUCCESS')}")
        print(f"Failed: {sum(1 for r in self.results if r.get('status') == 'FAILED')}")
        print(f"{'='*80}\n")
    
    def save_results(self, format='csv', filename=None):
        """Save results to CSV or Excel file"""
        if not self.results:
            print("No results to save!")
            return
        
        df = pd.DataFrame(self.results)
        
        if filename:
            output_file = self.output_dir / filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if format == 'csv':
                output_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
            elif format == 'excel':
                output_file = self.output_dir / f"benchmark_results_{timestamp}.xlsx"
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        if format == 'csv' or str(output_file).endswith('.csv'):
            df.to_csv(output_file, index=False)
        elif format == 'excel' or str(output_file).endswith('.xlsx'):
            df.to_excel(output_file, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"\nResults saved to: {output_file}")
        
        # Also print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        # Calculate summary statistics
        successful = df[df['status'] == 'SUCCESS']
        if len(successful) > 0:
            print(f"\nSuccessful runs: {len(successful)}/{len(df)}")
            
            # Try to convert to numeric for statistics
            try:
                avg_inf = pd.to_numeric(successful['avg_inference_ms'], errors='coerce')
                if not avg_inf.isna().all():
                    print(f"Average inference time: {avg_inf.mean():.3f} ms")
                    print(f"Min inference time: {avg_inf.min():.3f} ms")
                    print(f"Max inference time: {avg_inf.max():.3f} ms")
            except:
                pass
        
        print("\n" + df.to_string(index=False))
        print("="*80)
        
        return output_file


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Automate ONNX Runtime Web benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models with default settings
  python run_benchmark_cli.py --url http://localhost:5173 --models ./public/models/models.json
  
  # Run with specific backend and custom runs
  python run_benchmark_cli.py --url http://localhost:5173 --models ./public/models/models.json --backend webgpu --warmup 3 --measure 50
  
  # Run only models matching a filter
  python run_benchmark_cli.py --url http://localhost:5173 --models ./public/models/models.json --filter mobilenet resnet
  
  # Run in headless mode and save as Excel
  python run_benchmark_cli.py --url http://localhost:5173 --models ./public/models/models.json --headless --format excel
        """
    )
    
    parser.add_argument('--url', '-u', required=True,
                        help='URL of the benchmark web application (e.g., http://localhost:5173)')
    
    parser.add_argument('--models', '-m', required=True,
                        help='Path to models.json configuration file')
    
    parser.add_argument('--backend', '-b', default='wasm',
                        choices=['wasm', 'webgl', 'webgpu', 'webnn'],
                        help='Execution provider backend (default: wasm)')
    
    parser.add_argument('--warmup', '-w', type=int, default=1,
                        help='Number of warmup runs (default: 1)')
    
    parser.add_argument('--measure', '-r', type=int, default=100,
                        help='Number of measurement runs (default: 100)')
    
    parser.add_argument('--output-dir', '-o', default='benchmark_results',
                        help='Output directory for results (default: benchmark_results)')
    
    parser.add_argument('--format', '-f', default='csv',
                        choices=['csv', 'excel'],
                        help='Output format (default: csv)')
    
    parser.add_argument('--filename', '-n', default=None,
                        help='Custom output filename (default: auto-generated with timestamp)')
    
    parser.add_argument('--headless', action='store_true',
                        help='Run browser in headless mode')
    
    parser.add_argument('--filter', nargs='+', default=None,
                        help='Only run models matching these keywords (case-insensitive)')
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    print("="*80)
    print("ONNX Runtime Web Benchmark Automation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  URL: {args.url}")
    print(f"  Models config: {args.models}")
    print(f"  Backend: {args.backend}")
    print(f"  Warmup runs: {args.warmup}")
    print(f"  Measure runs: {args.measure}")
    print(f"  Output format: {args.format}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Headless: {args.headless}")
    if args.filter:
        print(f"  Model filter: {args.filter}")
    print("="*80)
    
    # Create runner
    runner = BenchmarkRunner(
        url=args.url,
        models_config_path=args.models,
        output_dir=args.output_dir,
        headless=args.headless
    )
    
    # Run benchmarks
    await runner.run_all_models(
        backend=args.backend,
        warmup=args.warmup,
        measure=args.measure,
        models_filter=args.filter
    )
    
    # Save results
    runner.save_results(format=args.format, filename=args.filename)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
