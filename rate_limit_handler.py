"""
Rate limit handler module for FBref scraper
This module provides tools to manage rate limiting when scraping FBref.com
"""

import time
import random
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("fbref_scraper")

class RateLimitHandler:
    """
    Class to handle rate limiting for web scraping operations.
    Implements adaptive delays, exponential backoff, and request tracking.
    """
    
    def __init__(self, min_delay=10, max_delay=20, max_requests_per_hour=25):
        """
        Initialize the rate limit handler
        
        Args:
            min_delay: Minimum delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
            max_requests_per_hour: Maximum number of requests per hour to allow
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_requests_per_hour = max_requests_per_hour
        self.request_history = []
        self.rate_limited_count = 0
        self.last_backoff_delay = min_delay
        
    def wait_before_request(self):
        """
        Calculate and wait for an appropriate time before making a request
        Implements adaptive delay based on recent rate limits
        
        Returns:
            The delay duration in seconds that was applied
        """
        # Clean old request history
        self._clean_history()
        
        # Calculate current hourly request count
        hourly_requests = len(self.request_history)
        
        # Adjust delay based on recent activity
        current_delay = self._calculate_delay(hourly_requests)
        
        # Add jitter (75-125% of calculated delay)
        jitter = random.random() * 0.5 + 0.75
        final_delay = current_delay * jitter
        
        logger.debug(f"Waiting {final_delay:.2f}s before next request (hourly: {hourly_requests}/{self.max_requests_per_hour})")
        time.sleep(final_delay)
        
        # Record this request
        self.request_history.append(datetime.now())
        
        return final_delay
    
    def _clean_history(self):
        """Remove requests older than 1 hour from history"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.request_history = [t for t in self.request_history if t > cutoff_time]
    
    def _calculate_delay(self, hourly_requests):
        """
        Calculate appropriate delay based on request volume and rate limit history
        
        Args:
            hourly_requests: Current number of requests in the past hour
            
        Returns:
            Calculated delay in seconds
        """
        # Base delay calculation
        ratio = hourly_requests / self.max_requests_per_hour
        
        if ratio < 0.5:
            # Low activity - use normal delay
            calculated_delay = self.min_delay
        elif ratio < 0.75:
            # Medium activity - scale between min and max
            scale_factor = (ratio - 0.5) * 4  # 0.0 to 1.0 scaling
            calculated_delay = self.min_delay + scale_factor * (self.max_delay - self.min_delay)
        else:
            # High activity - use max delay plus extra buffer
            extra = (ratio - 0.75) * 4 * self.max_delay  # Additional buffer beyond max_delay
            calculated_delay = self.max_delay + extra
        
        # Add penalty for recent rate limits
        rate_limit_penalty = self.rate_limited_count * self.min_delay
        
        return calculated_delay + rate_limit_penalty
    
    def handle_rate_limit(self, retry_count, max_retries):
        """
        Handle a rate limit response
        
        Args:
            retry_count: Current retry attempt number
            max_retries: Maximum number of retries allowed
            
        Returns:
            Tuple of (should_retry, delay_duration)
        """
        # Increment counter
        self.rate_limited_count += 1
        
        # Implement exponential backoff with jitter
        self.last_backoff_delay *= 2
        jitter = random.random() * 0.3 + 0.85  # 85-115% jitter
        backoff_time = self.last_backoff_delay * jitter
        
        # Decide if we should retry
        should_retry = retry_count < max_retries
        
        # Log the action
        if should_retry:
            logger.warning(f"Rate limited, retrying in {backoff_time:.2f} seconds (attempt {retry_count}/{max_retries})")
        else:
            logger.error(f"Maximum retries reached after {retry_count} attempts")
            # Reset backoff for next request
            self.last_backoff_delay = self.min_delay
        
        return (should_retry, backoff_time)
    
    def reset_after_success(self):
        """Reset backoff counter after a successful request"""
        if self.rate_limited_count > 0:
            self.rate_limited_count = max(0, self.rate_limited_count - 1)
            self.last_backoff_delay = self.min_delay