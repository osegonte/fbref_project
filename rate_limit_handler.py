"""
Enhanced rate limit handler module for FBref scraper
This module provides tools to manage rate limiting when scraping FBref.com
with improved resilience against 429 responses.
"""

import time
import random
import logging
import json
from datetime import datetime, timedelta
from rate_limit_handler import RateLimitHandler

logger = logging.getLogger("fbref_scraper")

class RateLimitHandler:
    """
    Class to handle rate limiting for web scraping operations.
    Implements adaptive delays, exponential backoff, and request tracking
    with improved avoidance strategies.
    """
    
    def __init__(self, min_delay=15, max_delay=30, max_requests_per_hour=20):
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
        self.hourly_quotas = {}  # Track quotas per hour
        self.current_hour = self._get_current_hour()
        
    def _get_current_hour(self):
        """Get the current hour as string in format YYYY-MM-DD-HH"""
        return datetime.now().strftime("%Y-%m-%d-%H")
        
    def wait_before_request(self):
        """
        Calculate and wait for an appropriate time before making a request
        Implements adaptive delay based on recent rate limits
        
        Returns:
            The delay duration in seconds that was applied
        """
        # Clean old request history
        self._clean_history()
        
        # Update hourly tracking
        hour_key = self._get_current_hour()
        if hour_key != self.current_hour:
            # Reset for new hour
            self.current_hour = hour_key
            
        if hour_key not in self.hourly_quotas:
            self.hourly_quotas[hour_key] = 0
            
        # Calculate current hourly request count
        hourly_requests = self.hourly_quotas[hour_key]
        
        # Adjust delay based on recent activity
        current_delay = self._calculate_delay(hourly_requests)
        
        # Add jitter (75-125% of calculated delay)
        jitter = random.random() * 0.5 + 0.75
        final_delay = current_delay * jitter
        
        logger.debug(f"Waiting {final_delay:.2f}s before next request (hourly: {hourly_requests}/{self.max_requests_per_hour})")
        time.sleep(final_delay)
        
        # Record this request
        self.request_history.append(datetime.now())
        self.hourly_quotas[hour_key] = hourly_requests + 1
        
        return final_delay
    
    def _clean_history(self):
        """Remove requests older than 1 hour from history"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.request_history = [t for t in self.request_history if t > cutoff_time]
        
        # Clean up old hour entries
        current_time = datetime.now()
        keys_to_remove = []
        for hour_key in self.hourly_quotas:
            try:
                hour_time = datetime.strptime(hour_key, "%Y-%m-%d-%H")
                if (current_time - hour_time) > timedelta(hours=2):
                    keys_to_remove.append(hour_key)
            except ValueError:
                keys_to_remove.append(hour_key)
                
        for key in keys_to_remove:
            del self.hourly_quotas[key]
    
    def _calculate_delay(self, hourly_requests):
        """
        Calculate appropriate delay based on request volume and rate limit history
        
        Args:
            hourly_requests: Current number of requests in the past hour
            
        Returns:
            Calculated delay in seconds
        """
        # Base delay calculation with exponential increase
        ratio = hourly_requests / self.max_requests_per_hour
        
        if ratio < 0.3:  # Less than 30% of limit
            calculated_delay = random.uniform(self.min_delay, self.min_delay * 1.5)
        elif ratio < 0.6:  # Between 30-60% of limit
            calculated_delay = random.uniform(self.min_delay * 1.5, self.max_delay)
        elif ratio < 0.8:  # Between 60-80% of limit
            calculated_delay = random.uniform(self.max_delay, self.max_delay * 2)
        else:  # Above 80% of limit
            calculated_delay = random.uniform(self.max_delay * 2, self.max_delay * 3)
        
        # Add penalty for recent rate limits
        rate_limit_penalty = self.rate_limited_count * self.min_delay
        
        # Apply time of day adjustment - slower at peak hours
        hour_of_day = datetime.now().hour
        time_of_day_factor = 1.0
        
        # Peak hours (9am-6pm) - add 20-50% more delay
        if 9 <= hour_of_day <= 18:
            time_of_day_factor = random.uniform(1.2, 1.5)
            
        return (calculated_delay + rate_limit_penalty) * time_of_day_factor
    
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
        
        # Apply a longer cooldown period if we're hitting limits repeatedly
        if self.rate_limited_count > 3:
            cooldown_multiplier = min(self.rate_limited_count, 10)
            backoff_time *= cooldown_multiplier / 3
            
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
            
    def save_state(self, filepath="data/rate_limit_state.json"):
        """Save current rate limiting state to a file"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "rate_limited_count": self.rate_limited_count,
            "hourly_quotas": self.hourly_quotas,
            "current_hour": self.current_hour
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f)
            logger.debug("Saved rate limit state")
            return True
        except Exception as e:
            logger.warning(f"Failed to save rate limit state: {e}")
            return False
    
    def load_state(self, filepath="data/rate_limit_state.json"):
        """Load rate limiting state from a file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.rate_limited_count = state.get("rate_limited_count", 0)
            self.hourly_quotas = state.get("hourly_quotas", {})
            self.current_hour = state.get("current_hour", self._get_current_hour())
            
            logger.info(f"Loaded rate limit state with {self.rate_limited_count} recent limits")
            return True
        except Exception as e:
            logger.debug(f"No previous rate limit state found or error loading: {e}")
            return False