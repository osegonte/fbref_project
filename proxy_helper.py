"""
Enhanced proxy helper for handling rotating proxies to avoid rate limiting
with additional connection testing and failure tracking
"""

import os
import random
import logging
import time
import requests
from dotenv import load_dotenv

logger = logging.getLogger("fbref_scraper")

class ProxyManager:
    """
    Class to manage a pool of proxies for web scraping
    
    Notes:
        - If no proxies are provided, requests will be made directly
        - To use proxies, set the PROXY_LIST environment variable with comma-separated proxies
        - Example: PROXY_LIST=http://user:pass@ip1:port,http://user:pass@ip2:port
    """
    
    def __init__(self, test_url="https://fbref.com/en/"):
        """
        Initialize the proxy manager
        
        Args:
            test_url: URL to use for testing proxy connection
        """
        # Load environment variables
        load_dotenv()
        
        # Parse proxy list from environment variable
        self.proxies = []
        self.test_url = test_url
        proxy_list_env = os.getenv('PROXY_LIST', '')
        
        if proxy_list_env:
            self.proxies = [p.strip() for p in proxy_list_env.split(',') if p.strip()]
            logger.info(f"Loaded {len(self.proxies)} proxies from environment")
            
            # Test all proxies on initialization
            self._test_all_proxies()
        
        self.last_proxy = None
        self.failed_proxies = set()
        self.proxy_stats = {}  # Track success/failure stats for each proxy
        
    def _test_all_proxies(self):
        """Test all proxies and remove non-working ones"""
        working_proxies = []
        
        for proxy in self.proxies:
            if self._test_proxy(proxy):
                working_proxies.append(proxy)
                self.proxy_stats[proxy] = {"success": 1, "failure": 0, "last_used": time.time()}
            else:
                logger.warning(f"Proxy {self._mask_proxy(proxy)} failed initial connection test")
                self.proxy_stats[proxy] = {"success": 0, "failure": 1, "last_used": time.time()}
                
        self.proxies = working_proxies
        logger.info(f"{len(self.proxies)} proxies passed the connection test")
    
    def _test_proxy(self, proxy_url):
        """
        Test if a proxy is working
        
        Args:
            proxy_url: Proxy URL to test
            
        Returns:
            True if proxy is working, False otherwise
        """
        try:
            proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml'
            }
            
            response = requests.get(
                self.test_url, 
                proxies=proxies,
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception:
            return False
    
    def _mask_proxy(self, proxy_url):
        """Mask the proxy URL to hide credentials in logs"""
        if '@' in proxy_url:
            parts = proxy_url.split('@')
            return f"...@{parts[1]}"
        return proxy_url
    
    def get_proxy(self):
        """
        Get a random proxy from the pool
        
        Returns:
            Dictionary with proxy configuration or None if no proxies available
        """
        # If no proxies or all failed, return None
        available_proxies = [p for p in self.proxies if p not in self.failed_proxies]
        
        if not available_proxies:
            # Try to recover some failed proxies if we've run out
            if self.failed_proxies and random.random() < 0.3:  # 30% chance to retry a failed proxy
                retry_proxy = random.choice(list(self.failed_proxies))
                logger.info(f"No available proxies, retrying previously failed proxy")
                self.failed_proxies.remove(retry_proxy)
                available_proxies = [retry_proxy]
            else:
                return None
        
        # Choose best proxy based on success rate and last used time
        if random.random() < 0.8:  # 80% of the time use weighted selection
            proxy = self._select_weighted_proxy(available_proxies)
        else:  # 20% of the time use random selection for exploration
            proxy = random.choice(available_proxies)
        
        # Update last used time
        if proxy in self.proxy_stats:
            self.proxy_stats[proxy]["last_used"] = time.time()
        
        self.last_proxy = proxy
        
        logger.debug(f"Using proxy: {self._mask_proxy(proxy)}")
        
        return {
            'http': proxy,
            'https': proxy
        }
    
    def _select_weighted_proxy(self, available_proxies):
        """
        Select a proxy using a weighted algorithm based on success rate and recency
        
        Args:
            available_proxies: List of available proxy URLs
            
        Returns:
            Selected proxy URL
        """
        now = time.time()
        weights = []
        
        for proxy in available_proxies:
            if proxy in self.proxy_stats:
                stats = self.proxy_stats[proxy]
                success = stats.get("success", 0)
                failure = stats.get("failure", 0)
                last_used = stats.get("last_used", 0)
                
                # Calculate success rate (default to 0.5 if no data)
                success_rate = 0.5
                if success + failure > 0:
                    success_rate = success / (success + failure)
                
                # Calculate recency factor (higher for less recently used proxies)
                recency = min(1.0, (now - last_used) / 3600)  # Scale based on hours since last use
                
                # Combined weight
                weight = success_rate * (0.7 + 0.3 * recency)
                weights.append(weight)
            else:
                # Default weight for proxies without stats
                weights.append(0.5)
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        else:
            weights = [1/len(available_proxies) for _ in available_proxies]
        
        # Weighted random selection
        return random.choices(available_proxies, weights=weights, k=1)[0]
    
    def mark_proxy_success(self, proxy_url):
        """
        Mark a proxy as successful
        
        Args:
            proxy_url: URL of the proxy that succeeded
        """
        if proxy_url:
            # Extract the base proxy URL if it's in a dictionary
            if isinstance(proxy_url, dict):
                proxy_url = proxy_url.get('https') or proxy_url.get('http')
                
            if proxy_url in self.proxies:
                if proxy_url in self.failed_proxies:
                    self.failed_proxies.remove(proxy_url)
                
                if proxy_url in self.proxy_stats:
                    self.proxy_stats[proxy_url]["success"] += 1
                else:
                    self.proxy_stats[proxy_url] = {"success": 1, "failure": 0, "last_used": time.time()}
    
    def mark_proxy_failed(self, proxy_url):
        """
        Mark a proxy as failed
        
        Args:
            proxy_url: URL of the proxy that failed
        """
        if proxy_url:
            # Extract the base proxy URL if it's in a dictionary
            if isinstance(proxy_url, dict):
                proxy_url = proxy_url.get('https') or proxy_url.get('http')
                
            if proxy_url in self.proxies:
                self.failed_proxies.add(proxy_url)
                
                if proxy_url in self.proxy_stats:
                    self.proxy_stats[proxy_url]["failure"] += 1
                else:
                    self.proxy_stats[proxy_url] = {"success": 0, "failure": 1, "last_used": time.time()}
                    
                logger.warning(f"Marked proxy as failed: {self._mask_proxy(proxy_url)}")
    
    def reset_failed_proxies(self):
        """Reset the list of failed proxies"""
        self.failed_proxies = set()
        logger.info("Reset failed proxy list")
    
    def has_proxies(self):
        """Check if any proxies are configured"""
        return len(self.proxies) > 0
        
    def get_stats(self):
        """Get statistics on proxy usage"""
        stats = {
            "total_proxies": len(self.proxies),
            "available_proxies": len(self.proxies) - len(self.failed_proxies),
            "failed_proxies": len(self.failed_proxies)
        }
        
        # Calculate success rates
        success_rates = {}
        for proxy, proxy_stats in self.proxy_stats.items():
            success = proxy_stats.get("success", 0)
            failure = proxy_stats.get("failure", 0)
            
            if success + failure > 0:
                success_rate = success / (success + failure)
                masked_proxy = self._mask_proxy(proxy)
                success_rates[masked_proxy] = {
                    "success_rate": f"{success_rate:.2%}",
                    "success": success,
                    "failure": failure
                }
        
        stats["proxy_success_rates"] = success_rates
        return stats