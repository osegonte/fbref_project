"""
Improved rate limit handler for FBref scraper
This module provides tools to manage rate limiting when scraping FBref.com
"""

import time
import random
import logging
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger("fbref_scraper")

class RateLimitHandler:
    """
    Class to handle rate limiting for web scraping operations.
    Implements adaptive delays, exponential backoff, and domain-level cooldown
    """
    
    def __init__(self, min_delay=10, max_delay=20, cooldown_threshold=3):
        """
        Initialize the rate limit handler
        
        Args:
            min_delay: Minimum delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
            cooldown_threshold: Number of rate limits before enforcing a full domain cooldown
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.cooldown_threshold = cooldown_threshold
        self.rate_limited_count = 0
        self.last_request_time = 0
        
        # Domain-level cooldown tracker
        self.domain_cooldown_until = 0
        
        # Request history for adaptive pacing
        self.request_history = []
        self.rate_limit_history = []
        
        # Create cache directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
    def wait_before_request(self):
        """
        Determine and apply appropriate wait time before making a request
        """
        current_time = time.time()
        
        # Check for domain-level cooldown
        if current_time < self.domain_cooldown_until:
            cooldown_wait = self.domain_cooldown_until - current_time
            logger.info(f"Domain cooldown in effect, waiting {cooldown_wait:.2f}s")
            time.sleep(cooldown_wait)
            current_time = time.time()
        
        # Calculate delay based on FBref's robots.txt crawl-delay (10s) plus jitter
        base_delay = 10  # FBref's crawl-delay
        
        # Add jitter (±20%)
        jitter_factor = random.uniform(0.8, 1.2)
        delay = base_delay * jitter_factor
        
        # Add penalty for recent rate limits
        if self.rate_limited_count > 0:
            penalty = min(self.rate_limited_count * 5, 30)  # Cap at 30s extra
            delay += penalty
        
        # Ensure minimum delay since last request
        elapsed = current_time - self.last_request_time
        if elapsed < delay:
            wait_time = delay - elapsed
            logger.debug(f"Waiting {wait_time:.2f}s before next request")
            time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time = time.time()
        self.request_history.append(self.last_request_time)
        
        # Clean old history
        self._clean_history()
        
        return delay
    
    def _clean_history(self):
        """Remove request records older than 1 hour"""
        cutoff_time = time.time() - 3600
        self.request_history = [t for t in self.request_history if t > cutoff_time]
        self.rate_limit_history = [t for t in self.rate_limit_history if t > cutoff_time]
    
    def handle_rate_limit(self, retry_count, max_retries):
        """
        Handle a rate limit response (429)
        
        Args:
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries allowed
            
        Returns:
            Tuple of (should_retry, backoff_time)
        """
        current_time = time.time()
        self.rate_limited_count += 1
        self.rate_limit_history.append(current_time)
        
        # Check if we need to implement a domain-level cooldown
        recent_rate_limits = len([t for t in self.rate_limit_history if t > current_time - 300])
        
        if recent_rate_limits >= self.cooldown_threshold:
            # Implement a long domain-level cooldown (60-120 seconds)
            cooldown_time = random.uniform(60, 120)
            self.domain_cooldown_until = current_time + cooldown_time
            logger.warning(f"Too many rate limits ({recent_rate_limits} in 5 min). Domain cooldown for {cooldown_time:.2f}s")
        
        # Calculate backoff time with exponential increase
        # Start with 2s, then 4s, 8s, 16s, 32s, 64s
        backoff_time = min(2 ** retry_count, 64)
        
        # Add jitter (±30%)
        jitter = random.uniform(0.7, 1.3)
        backoff_time *= jitter
        
        # Determine if we should retry
        should_retry = retry_count < max_retries
        
        if should_retry:
            logger.warning(f"Rate limited, retrying in {backoff_time:.2f} seconds (attempt {retry_count}/{max_retries})")
        else:
            logger.error(f"Maximum retries reached ({max_retries})")
        
        return should_retry, backoff_time
    
    def reset_after_success(self):
        """Gradually reduce rate limit counter after successful requests"""
        # Gradually reduce rate limited count
        if self.rate_limited_count > 0:
            self.rate_limited_count = max(0, self.rate_limited_count - 0.5)
    
    def save_state(self, filepath="data/rate_limit_state.json"):
        """Save current rate limiting state to file"""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "rate_limited_count": self.rate_limited_count,
                "domain_cooldown_until": self.domain_cooldown_until,
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f)
            return True
        except Exception as e:
            logger.warning(f"Failed to save rate limit state: {e}")
            return False
    
    def load_state(self, filepath="data/rate_limit_state.json"):
        """Load previous rate limiting state from file"""
        try:
            if not os.path.exists(filepath):
                return False
                
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.rate_limited_count = state.get("rate_limited_count", 0)
            self.domain_cooldown_until = state.get("domain_cooldown_until", 0)
            
            # Check if cooldown is still valid
            if self.domain_cooldown_until > time.time():
                cooldown_remaining = self.domain_cooldown_until - time.time()
                logger.info(f"Loaded existing cooldown, {cooldown_remaining:.2f}s remaining")
            
            return True
        except Exception as e:
            logger.debug(f"No previous rate limit state loaded: {e}")
            return False