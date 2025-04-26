"""
Enhanced polite request module for FBref scraping
Implements browser-like behavior and proper rate-limit handling
"""

import requests
import time
import random
import logging
import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple

logger = logging.getLogger("fbref_scraper")

# Expanded list of User Agents to better mimic real browsers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.34',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/111.0'
]

class PoliteRequester:
    """
    Class to make polite requests to FBref respecting rate limits
    with robust caching, user-agent rotation, and cookie persistence
    """
    
    def __init__(self, 
                 cache_dir: str = "data/cache", 
                 max_cache_age: int = 168,  # 1 week cache
                 min_delay: int = 10,  # Respect FBref's robots.txt
                 session = None):
        """
        Initialize the polite requester
        
        Args:
            cache_dir: Directory to store cached responses
            max_cache_age: Maximum age of cached responses in hours
            min_delay: Minimum delay between requests
            session: Optional existing requests.Session
        """
        self.cache_dir = cache_dir
        self.max_cache_age = max_cache_age  # Hours
        self.min_delay = min_delay
        
        # Create a persistent session
        self.session = session or requests.Session()
        
        # Domain-level cooldown tracking
        self.domain_cooldown_until = 0
        
        # Request tracking
        self.last_request_time = 0
        self.consecutive_failures = 0
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cached_response(self, url: str) -> Optional[str]:
        """
        Get cached response for a URL if available and not expired
        
        Args:
            url: URL to check for cached response
            
        Returns:
            Cached HTML content or None if not cached or expired
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{url_hash}.html")
        
        if os.path.exists(cache_file):
            # Check age of cache file
            file_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            
            if file_age_hours < self.max_cache_age:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to read cache file for {url}: {e}")
        
        return None
    
    def save_to_cache(self, url: str, content: str) -> bool:
        """
        Save response content to cache
        
        Args:
            url: URL associated with the content
            content: HTML content to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{url_hash}.html")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
            return False
    
    def get_random_headers(self) -> Dict[str, str]:
        """
        Get random browser-like headers
        
        Returns:
            Dictionary of headers
        """
        user_agent = random.choice(USER_AGENTS)
        
        # Standard browser headers
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'TE': 'Trailers',
            'DNT': '1'
        }
        
        return headers
    
    def wait_for_rate_limit(self) -> float:
        """
        Wait an appropriate amount of time before making a request
        
        Returns:
            The actual delay applied in seconds
        """
        current_time = time.time()
        
        # Check for domain-level cooldown
        if current_time < self.domain_cooldown_until:
            cooldown_wait = self.domain_cooldown_until - current_time
            logger.info(f"Respecting domain cooldown, waiting {cooldown_wait:.2f}s")
            time.sleep(cooldown_wait + random.uniform(1, 3))  # Add a bit extra
            current_time = time.time()
        
        # Calculate basic delay (10s recommended in FBref's robots.txt)
        base_delay = self.min_delay
        
        # Add randomness (±20%)
        jitter = random.uniform(0.8, 1.2)
        delay = base_delay * jitter
        
        # Add penalty for consecutive failures
        if self.consecutive_failures > 0:
            penalty = min(self.consecutive_failures * 5, 30)  # Maximum 30s extra
            delay += penalty
        
        # Ensure minimum time since last request
        time_since_last = current_time - self.last_request_time
        if time_since_last < delay:
            wait_time = delay - time_since_last
            logger.debug(f"Waiting {wait_time:.2f}s before request")
            time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time = time.time()
        
        return delay
    
    def fetch(self, url: str, use_cache: bool = True, max_retries: int = 5) -> Optional[str]:
        """
        Fetch URL content with caching, rate-limiting, and retry logic
        
        Args:
            url: URL to fetch
            use_cache: Whether to use/update cache
            max_retries: Maximum number of retry attempts
            
        Returns:
            HTML content as string or None if failed
        """
        # Check cache first if enabled
        if use_cache:
            cached_content = self.get_cached_response(url)
            if cached_content:
                logger.debug(f"Using cached version of {url}")
                self.consecutive_failures = 0  # Reset failure counter on cache hit
                return cached_content
        
        # Apply rate limit delay
        self.wait_for_rate_limit()
        
        # Set up for request
        headers = self.get_random_headers()
        
        # Add referrer to look more natural
        if random.random() < 0.8:  # 80% chance to add referrer
            # Make it look like internal navigation
            parts = url.split("/")
            if len(parts) > 3:
                base_url = "/".join(parts[:3])
                possible_referrers = [
                    f"{base_url}/",
                    f"{base_url}/en/",
                    f"{base_url}/en/comps/"
                ]
                headers['Referer'] = random.choice(possible_referrers)
        
        # Make request with retries
        for attempt in range(max_retries):
            try:
                logger.debug(f"Requesting {url} (attempt {attempt+1}/{max_retries})")
                
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=30,
                    allow_redirects=True
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    self.consecutive_failures += 1
                    
                    # Calculate backoff time - exponential with jitter
                    backoff = min(2 ** attempt * (1 + random.random() * 0.3), 60)
                    
                    # Set domain cooldown if we've been rate limited multiple times
                    if attempt >= 2:
                        cooldown = 60 + random.uniform(0, 30)  # 60-90 seconds
                        self.domain_cooldown_until = time.time() + cooldown
                        logger.warning(f"Rate limited multiple times, setting domain cooldown for {cooldown:.2f}s")
                    
                    logger.warning(f"Rate limited (429), backing off for {backoff:.2f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                    continue
                
                # For other errors, fail fast or retry with backoff
                response.raise_for_status()
                
                # Request succeeded
                self.consecutive_failures = 0  # Reset on success
                
                # Cache the successful response
                if use_cache:
                    self.save_to_cache(url, response.text)
                
                return response.text
                
            except requests.exceptions.RequestException as e:
                self.consecutive_failures += 1
                
                # Determine if we should retry
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt * (1 + random.random() * 0.3)
                    logger.warning(f"Request failed: {e}. Retrying in {backoff:.2f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                else:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                    return None
        
        return None