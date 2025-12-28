import asyncio
import json
import re
import time
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright
import pandas as pd


class BenchmarkRunner:
    def __init__(self, url, models_config_path, output_dir="benchmark_results"):
        self.url = url
        self.models_config_path = models_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
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
            await page.wait_for_selector("#runBtn", state="visible", timeout=30000)
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
            
            while True:
                # Check if run button is enabled (benchmark complete)
                is_disabled = await page.locator("#runBtn").is_disabled()
                
                if not is_disabled:
                    print("Benchmark completed!")
                    break
                
                # Check timeout
                if (time.time() - start_time) * 1000 > timeout_ms:
                    raise Exception(f"Benchmark timed out after {timeout_ms/1000} seconds")
                
                # Log progress every 5 seconds
                if int(time.time() - start_time) % 5 == 0:
                    status_text = await page.locator("#status").text_content()
                    last_lines = status_text.strip().split('\n')[-3:]
                    print(f"Progress: {' | '.join(last_lines)}")
                
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
            'notes': ''
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
    
    async def run_all_models(self, backend="wasm", warmup=1, measure=100):
        """Run benchmarks for all models"""
        models = await self.load_models_list()
        
        print(f"\n{'='*80}")
        print(f"Starting benchmark run for {len(models)} models")
        print(f"Backend: {backend}, Warmup: {warmup}, Measure: {measure}")
        print(f"{'='*80}\n")
        
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                headless=False,  # Set to True for headless mode
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
        print(f"{'='*80}\n")
    
    def save_results(self, format='csv'):
        """Save results to CSV or Excel file"""
        if not self.results:
            print("No results to save!")
            return
        
        df = pd.DataFrame(self.results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            output_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        elif format == 'excel':
            output_file = self.output_dir / f"benchmark_results_{timestamp}.xlsx"
            df.to_excel(output_file, index=False, engine='openpyxl')
            print(f"\nResults saved to: {output_file}")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Also print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return output_file


async def main():
    """Main entry point"""
    
    # Configuration
    BENCHMARK_URL = "http://localhost:5173"  # Adjust to your dev server URL
    MODELS_CONFIG = "./public/models/models.json"  # Adjust path to your models.json
    BACKEND = "webgl"  # Options: wasm, webgl, webgpu, webnn
    WARMUP_RUNS = 1
    MEASURE_RUNS = 100
    OUTPUT_FORMAT = "csv"  # Options: csv, excel
    
    # Create runner
    runner = BenchmarkRunner(BENCHMARK_URL, MODELS_CONFIG)
    
    # Run benchmarks
    await runner.run_all_models(
        backend=BACKEND,
        warmup=WARMUP_RUNS,
        measure=MEASURE_RUNS
    )
    
    # Save results
    runner.save_results(format=OUTPUT_FORMAT)


if __name__ == "__main__":
    print("="*80)
    print("ONNX Runtime Web Benchmark Automation")
    print("="*80)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()