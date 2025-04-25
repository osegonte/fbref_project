"""HTTP utilities for FBref Toolkit with improved rate limiting and caching."""
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union

import requests
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry

from config import (
    CACHE_DIR_PATH, USER_AGENT, 
    REQUEST_DELAY, CACHE_TTL, 
    MAX_REQUESTS_PER_MINUTE, MAX_RETRIES, BACKOFF_FACTOR
)

# Configure logging
logger = logging.getLogger("fbref_toolkit.http")

# Create cache directory if it doesn't exist
CACHE_DIR_PATH.mkdir(exist_ok=True, parents=True)

# Default headers for all requests
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml",
    "Accept-Language": "en-US,en;q=0.9",
}

class HTTPError(Exception):
    """Custom exception for HTTP-related errors."""
    pass

def _cache_path(url: str) -> Path:
    """Generate a cache file path from a URL.
    
    Args:
        url: The URL to generate a cache path for
        
    Returns:
        Path object for the cache file
    """
    # Create a more unique cache filename based on URL components
    url_parts = url.replace("https://", "").replace("http://", "").split("/")
    cache_name = "_".join(url_parts[-3:] if len(url_parts) >= 3 else url_parts)
    # Replace invalid filename characters
    cache_name = "".join(c if c.isalnum() or c in "_-." else "_" for c in cache_name)
    return CACHE_DIR_PATH / f"{cache_name}.html"

def _is_cache_valid(path: Path, ttl: int) -> bool:
    """Check if a cache file is still valid based on TTL.
    
    Args:
        path: Path to the cache file
        ttl: Time-to-live in seconds
        
    Returns:
        True if cache is valid, False otherwise
    """
    if not path.exists():
        return False
    
    # Check cache age
    cache_age = time.time() - path.stat().st_mtime
    return cache_age < ttl

@sleep_and_retry
@limits(calls=MAX_REQUESTS_PER_MINUTE, period=60)
def _rate_limited_request(url: str, headers: Dict[str, str], timeout: int) -> requests.Response:
    """Make a rate-limited HTTP request.
    
    Args:
        url: URL to fetch
        headers: Request headers
        timeout: Request timeout in seconds
        
    Returns:
        Response object
        
    Raises:
        HTTPError: If the request fails
    """
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        
        # Handle status codes
        if response.status_code == 429:
            # Too many requests - parse retry header if available
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    wait_time = int(retry_after)
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds as requested by server")
                    time.sleep(wait_time)
                    # Recursive call after waiting
                    return _rate_limited_request(url, headers, timeout)
                except ValueError:
                    # If retry-after isn't a valid integer, use default backoff
                    pass
        
        # Raise an error for other non-200 status codes
        response.raise_for_status()
        return response
        
    except requests.exceptions.RequestException as e:
        raise HTTPError(f"Request failed: {str(e)}")

def fetch(
    url: str, 
    use_cache: bool = True, 
    cache_ttl: int = CACHE_TTL, 
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Fetch URL content with caching and rate limiting.
    
    Args:
        url: The URL to fetch
        use_cache: Whether to use cached responses
        cache_ttl: Cache time-to-live in seconds
        timeout: Request timeout in seconds
        headers: Optional additional headers to include
        
    Returns:
        HTML content as string
        
    Raises:
        HTTPError: If the request fails after retries
    """
    path = _cache_path(url)
    
    # Check for valid cache
    if use_cache and _is_cache_valid(path, cache_ttl):
        logger.debug(f"Using cached response for {url}")
        return path.read_text(encoding="utf-8")
    
    # Merge default headers with any provided headers
    request_headers = HEADERS.copy()
    if headers:
        request_headers.update(headers)
    
    # Fetch with retry logic
    for attempt in range(MAX_RETRIES):
        try:
            # Exponential backoff between retries
            if attempt > 0:
                sleep_time = REQUEST_DELAY * (BACKOFF_FACTOR ** attempt)
                logger.debug(f"Retry {attempt+1}/{MAX_RETRIES}, waiting {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
            logger.debug(f"Fetching {url} (attempt {attempt + 1}/{MAX_RETRIES})")
            
            response = _rate_limited_request(url, request_headers, timeout)
            html = response.text
            
            # Write to cache
            path.write_text(html, encoding="utf-8")
            
            # Record last-modified and etag headers if present for future conditional requests
            headers_cache = {}
            for header in ['last-modified', 'etag']:
                if header in response.headers:
                    headers_cache[header] = response.headers[header]
            
            if headers_cache:
                headers_path = path.with_suffix('.headers')
                with open(headers_path, 'w') as f:
                    for k, v in headers_cache.items():
                        f.write(f"{k}: {v}\n")
                        
            # Log cache info
            logger.debug(f"Cached response for {url} at {path}")
            
            return html
            
        except HTTPError as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
    
    # If we get here, all attempts failed
    raise HTTPError(f"Failed to fetch {url} after {MAX_RETRIES} attempts")

def soupify(html: str) -> BeautifulSoup:
    """Convert HTML string to BeautifulSoup object.
    
    Args:
        html: HTML content as string
        
    Returns:
        BeautifulSoup object
    """
    return BeautifulSoup(html, "html.parser")

def clear_cache(days_old: Optional[int] = None) -> int:
    """Clear cached responses, optionally keeping recent ones.
    
    Args:
        days_old: Only clear cache files older than this many days
                  If None, clears all cache files
    
    Returns:
        Number of files deleted
    """
    deleted = 0
    now = time.time()
    
    for cache_file in CACHE_DIR_PATH.glob("*.html"):
        if days_old is not None:
            file_age_days = (now - cache_file.stat().st_mtime) / (60 * 60 * 24)
            if file_age_days < days_old:
                continue
                
        # Also delete corresponding headers file if it exists
        headers_file = cache_file.with_suffix('.headers')
        if headers_file.exists():
            headers_file.unlink()
        
        cache_file.unlink()
        deleted += 1
    
    logger.info(f"Cleared {deleted} cache files")
    return deleted