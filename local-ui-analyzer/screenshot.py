"""
Website Screenshot Capture Module
Captures full-page and viewport screenshots from URLs using Playwright.
"""

import asyncio
from typing import Tuple, Optional
from dataclasses import dataclass
from playwright.async_api import async_playwright


@dataclass
class ViewportConfig:
    """Viewport configuration for screenshot capture."""
    width: int
    height: int
    device_type: str  # 'mobile', 'tablet', 'desktop'
    is_mobile: bool = False
    device_scale_factor: float = 1.0


# Common viewport presets
VIEWPORT_PRESETS = {
    'mobile': ViewportConfig(375, 667, 'mobile', is_mobile=True, device_scale_factor=2),
    'mobile_landscape': ViewportConfig(667, 375, 'mobile', is_mobile=True, device_scale_factor=2),
    'tablet': ViewportConfig(768, 1024, 'tablet', device_scale_factor=2),
    'tablet_landscape': ViewportConfig(1024, 768, 'tablet', device_scale_factor=2),
    'desktop': ViewportConfig(1920, 1080, 'desktop'),
    'desktop_small': ViewportConfig(1366, 768, 'desktop'),
    'desktop_large': ViewportConfig(2560, 1440, 'desktop'),
}


def detect_viewport_type(width: int, height: int) -> str:
    """
    Classify viewport as mobile, tablet, or desktop based on width.
    
    Args:
        width: Viewport width in pixels
        height: Viewport height in pixels
        
    Returns:
        'mobile', 'tablet', or 'desktop'
    """
    if width < 768:
        return 'mobile'
    elif width < 1024:
        return 'tablet'
    else:
        return 'desktop'


def get_viewport_config(viewport: str = 'desktop', 
                        custom_size: Optional[Tuple[int, int]] = None) -> ViewportConfig:
    """
    Get viewport configuration.
    
    Args:
        viewport: Preset name ('mobile', 'tablet', 'desktop') or 'custom'
        custom_size: Tuple of (width, height) for custom viewport
        
    Returns:
        ViewportConfig object
    """
    if custom_size:
        width, height = custom_size
        device_type = detect_viewport_type(width, height)
        return ViewportConfig(
            width=width,
            height=height,
            device_type=device_type,
            is_mobile=(device_type == 'mobile'),
            device_scale_factor=2 if device_type in ('mobile', 'tablet') else 1
        )
    
    return VIEWPORT_PRESETS.get(viewport, VIEWPORT_PRESETS['desktop'])


async def _capture_website_async(
    url: str,
    viewport_config: ViewportConfig,
    output_path: str,
    full_page: bool = True,
    wait_time: int = 2000
) -> dict:
    """
    Async implementation of website screenshot capture.
    
    Returns:
        Dict with screenshot info including paths and page dimensions
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        context = await browser.new_context(
            viewport={'width': viewport_config.width, 'height': viewport_config.height},
            device_scale_factor=viewport_config.device_scale_factor,
            is_mobile=viewport_config.is_mobile,
            has_touch=viewport_config.is_mobile,
        )
        
        page = await context.new_page()
        
        # Navigate to URL with robust fallback strategy
        # Use 'domcontentloaded' instead of 'networkidle' - SPAs like Linear never reach networkidle
        # due to persistent WebSocket connections and background polling
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
        except Exception as e:
            # If domcontentloaded fails, try with 'load' as final fallback
            print(f"Warning: Initial navigation approach failed, retrying with 'load': {e}")
            await page.goto(url, wait_until='load', timeout=60000)
        
        # Wait for initial rendering and dynamic content to settle
        await page.wait_for_timeout(wait_time + 1000)
        
        # Scroll to bottom and back to trigger lazy loading
        await page.evaluate('''
            async () => {
                const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
                const distance = 500;
                const height = document.documentElement.scrollHeight;
                
                // Scroll down
                for (let i = 0; i < height; i += distance) {
                    window.scrollTo(0, i);
                    await delay(100);
                }
                
                // Scroll back to top
                window.scrollTo(0, 0);
                await delay(500);
            }
        ''')
        
        # Wait for lazy-loaded content to render
        await page.wait_for_timeout(1000)
        
        # Get page dimensions
        page_height = await page.evaluate('document.documentElement.scrollHeight')
        page_width = await page.evaluate('document.documentElement.scrollWidth')
        
        # Take viewport screenshot (above the fold)
        viewport_path = output_path.replace('.png', '_viewport.png')
        await page.screenshot(path=viewport_path, full_page=False)
        
        # Take full page screenshot
        fullpage_path = output_path.replace('.png', '_fullpage.png')
        await page.screenshot(path=fullpage_path, full_page=True)
        
        # Get page title
        title = await page.title()
        
        await browser.close()
        
        return {
            'viewport_screenshot': viewport_path,
            'fullpage_screenshot': fullpage_path,
            'viewport_width': viewport_config.width,
            'viewport_height': viewport_config.height,
            'page_width': page_width,
            'page_height': page_height,
            'device_type': viewport_config.device_type,
            'is_mobile': viewport_config.is_mobile,
            'page_title': title,
            'url': url,
            'fold_line_y': viewport_config.height,  # Above-the-fold boundary
        }


def capture_website(
    url: str,
    viewport: str = 'desktop',
    custom_size: Optional[Tuple[int, int]] = None,
    output_path: str = 'screenshot.png',
    wait_time: int = 2000
) -> dict:
    """
    Capture screenshots from a website URL.
    
    Args:
        url: Website URL to capture
        viewport: Preset ('mobile', 'tablet', 'desktop')
        custom_size: Custom (width, height) tuple
        output_path: Base output path for screenshots
        wait_time: Time to wait after page load (ms)
        
    Returns:
        Dict with screenshot paths and page info
    """
    viewport_config = get_viewport_config(viewport, custom_size)
    
    # Run async capture
    return asyncio.run(_capture_website_async(
        url, viewport_config, output_path, wait_time=wait_time
    ))


def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    return path.startswith(('http://', 'https://'))


if __name__ == "__main__":
    # Test capture
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
        viewport = sys.argv[2] if len(sys.argv) > 2 else 'desktop'
        
        print(f"Capturing {url} with {viewport} viewport...")
        result = capture_website(url, viewport, output_path='test_screenshot.png')
        
        print(f"\nResults:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("Usage: python screenshot.py <url> [viewport]")
        print("Viewports: mobile, tablet, desktop")
