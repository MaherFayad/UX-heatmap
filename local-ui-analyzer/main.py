"""
Local UI Analyzer - CLI Interface
Run UI/UX analysis on images or website URLs and generate interactive HTML reports.
"""

import os
import sys
import json
import argparse
import webbrowser
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader

# Import our modules
from analyze import run_analysis
from screenshot import capture_website, is_url, get_viewport_config


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_viewport(viewport_str: str) -> tuple:
    """Parse viewport string like '1920x1080' into (width, height) tuple."""
    try:
        parts = viewport_str.lower().split('x')
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        raise ValueError(f"Invalid viewport format: {viewport_str}. Use WIDTHxHEIGHT (e.g., 1920x1080)")


def generate_report(results: dict, output_path: str) -> str:
    """Generate HTML report from analysis results."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("report.html")
    
    html_content = template.render(**results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def output_json(results: dict, output_path: str = None) -> dict:
    """
    Output analysis results as JSON for Claude integration.
    Excludes base64 images for smaller output.
    """
    # Create a serializable copy without base64 images
    json_results = {
        'image_path': results.get('image_path'),
        'dimensions': results.get('dimensions'),
        'device_type': results.get('device_type'),
        'viewport_height': results.get('viewport_height'),
        'focus_score': results.get('focus_score'),
        'boxes': results.get('boxes', []),
        'above_fold_analysis': results.get('above_fold_analysis'),
        'scroll_analysis': results.get('scroll_analysis'),
        'accessibility_report': results.get('accessibility_report'),
        'page_info': results.get('page_info'),
    }
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, cls=NumpyEncoder)
    
    return json_results


def run_analysis_for_claude(
    input_path: str,
    viewport: str = 'desktop',
    custom_viewport: tuple = None,
    output_dir: str = 'output'
) -> dict:
    """
    Run analysis and return structured JSON results.
    This function can be called programmatically for Claude integration.
    
    Args:
        input_path: URL or local image path
        viewport: 'mobile', 'tablet', or 'desktop'
        custom_viewport: Optional (width, height) tuple
        output_dir: Directory for output files
        
    Returns:
        Analysis results as dict (suitable for JSON)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if is_url(input_path):
        # Capture website
        viewport_config = get_viewport_config(viewport, custom_viewport)
        screenshot_path = os.path.join(output_dir, "screenshot.png")
        page_info = capture_website(
            input_path,
            viewport=viewport,
            custom_size=custom_viewport,
            output_path=screenshot_path,
            wait_time=2000
        )
        image_path = page_info['fullpage_screenshot']
        viewport_height = page_info['viewport_height']
        device_type = page_info['device_type']
    else:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Image not found: {input_path}")
        image_path = input_path
        page_info = None
        if custom_viewport:
            viewport_height = custom_viewport[1]
            device_type = get_viewport_config('custom', custom_viewport).device_type
        elif viewport == 'mobile':
            viewport_height = 667
            device_type = 'mobile'
        elif viewport == 'tablet':
            viewport_height = 1024
            device_type = 'tablet'
        else:
            viewport_height = None
            device_type = None
    
    # Run analysis
    results = run_analysis(
        image_path,
        output_dir,
        viewport_height=viewport_height,
        device_type=device_type,
        page_info=page_info
    )
    
    # Return JSON-serializable results
    return output_json(results)


def main():
    parser = argparse.ArgumentParser(
        description="Local UI Analyzer - Attention Insight Clone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a local image
  python main.py screenshot.png
  
  # Analyze a website (desktop viewport)
  python main.py https://example.com
  
  # Analyze a website with mobile viewport
  python main.py https://example.com --mobile
  
  # Output JSON for Claude integration
  python main.py https://example.com --json
  
  # Save JSON to file
  python main.py https://example.com --json -o results.json
        """
    )
    
    parser.add_argument(
        "input",
        help="Path to image file or website URL to analyze"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="report.html",
        help="Output filename (default: report.html, or results.json with --json)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files (default: output)"
    )
    
    # Viewport options (mutually exclusive)
    viewport_group = parser.add_mutually_exclusive_group()
    viewport_group.add_argument(
        "--mobile",
        action="store_true",
        help="Use mobile viewport (375x667)"
    )
    viewport_group.add_argument(
        "--tablet",
        action="store_true",
        help="Use tablet viewport (768x1024)"
    )
    viewport_group.add_argument(
        "--desktop",
        action="store_true",
        help="Use desktop viewport (1920x1080) - default for URLs"
    )
    viewport_group.add_argument(
        "--viewport",
        metavar="WxH",
        help="Custom viewport size (e.g., 1440x900)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of HTML report (for Claude integration)"
    )
    
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't automatically open the report in browser"
    )
    
    parser.add_argument(
        "--wait",
        type=int,
        default=2000,
        help="Time to wait after page load in ms (default: 2000)"
    )
    
    args = parser.parse_args()
    
    # Determine viewport
    viewport_preset = 'desktop'
    custom_viewport = None
    
    if args.mobile:
        viewport_preset = 'mobile'
    elif args.tablet:
        viewport_preset = 'tablet'
    elif args.viewport:
        custom_viewport = parse_viewport(args.viewport)
    
    # JSON output mode - minimal output for Claude
    if args.json:
        try:
            json_results = run_analysis_for_claude(
                args.input,
                viewport=viewport_preset,
                custom_viewport=custom_viewport,
                output_dir=args.output_dir
            )
            
            # Output to file or stdout
            if args.output and args.output != "report.html":
                json_path = os.path.join(args.output_dir, args.output)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_results, f, indent=2, cls=NumpyEncoder)
                print(f"JSON saved to: {json_path}", file=sys.stderr)
            else:
                # Print to stdout for piping
                print(json.dumps(json_results, indent=2, cls=NumpyEncoder))
            
            return
            
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)
    
    # Normal HTML report mode
    print("=" * 60)
    print("  🎯 Local UI Analyzer - Attention Insight Clone")
    print("=" * 60)
    print()
    
    try:
        # Check if input is a URL or image file
        if is_url(args.input):
            print(f"📸 Capturing website: {args.input}")
            
            viewport_config = get_viewport_config(viewport_preset, custom_viewport)
            print(f"📱 Viewport: {viewport_config.width}x{viewport_config.height} ({viewport_config.device_type})")
            
            os.makedirs(args.output_dir, exist_ok=True)
            
            screenshot_path = os.path.join(args.output_dir, "screenshot.png")
            page_info = capture_website(
                args.input,
                viewport=viewport_preset,
                custom_size=custom_viewport,
                output_path=screenshot_path,
                wait_time=args.wait
            )
            
            print(f"✅ Captured: {page_info['page_title']}")
            print(f"   Page size: {page_info['page_width']}x{page_info['page_height']}")
            
            image_path = page_info['fullpage_screenshot']
            viewport_height = page_info['viewport_height']
            device_type = page_info['device_type']
            
        else:
            if not os.path.exists(args.input):
                print(f"Error: Image not found: {args.input}")
                sys.exit(1)
            
            image_path = args.input
            page_info = None
            
            if custom_viewport:
                viewport_height = custom_viewport[1]
                device_type = get_viewport_config('custom', custom_viewport).device_type
            elif args.mobile:
                viewport_height = 667
                device_type = 'mobile'
            elif args.tablet:
                viewport_height = 1024
                device_type = 'tablet'
            else:
                viewport_height = None
                device_type = None
        
        # Run analysis
        results = run_analysis(
            image_path,
            args.output_dir,
            viewport_height=viewport_height,
            device_type=device_type,
            page_info=page_info
        )
        
        # Generate HTML report
        report_path = os.path.join(args.output_dir, args.output)
        generate_report(results, report_path)
        
        # Also save JSON for reference
        json_path = os.path.join(args.output_dir, "results.json")
        output_json(results, json_path)
        
        print()
        print("=" * 60)
        print(f"  ✅ Report generated: {report_path}")
        print(f"  📊 Focus Score: {results['focus_score']:.1f}%")
        print(f"  📱 Device Type: {results['device_type']}")
        print(f"  📐 Above Fold: {results['above_fold_analysis']['above_fold_attention_pct']:.1f}% attention")
        print(f"  🔢 Attention Areas: {len(results['boxes'])}")
        print("=" * 60)
        
        if not args.no_open:
            abs_path = os.path.abspath(report_path)
            print(f"\n🌐 Opening report in browser...")
            webbrowser.open(f"file:///{abs_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
