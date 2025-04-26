"""
Proxy helper for handling rotating proxies to avoid rate limiting
"""

import os
import random
import logging
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
    
    def __init__(self):
        """Initialize the proxy manager"""
        # Load environment variables
        load_dotenv()
        
        # Parse proxy list from environment variable
        self.proxies = []
        proxy_list_env = os.getenv('PROXY_LIST', '')
        
        if proxy_list_env:
            self.proxies = [p.strip() for p in proxy_list_env.split(',') if p.strip()]
            logger.info(f"Loaded {len(self.proxies)} proxies from environment")
        
        self.last_proxy = None
        self.failed_proxies = set()
    
    def get_proxy(self):
        """
        Get a random proxy from the pool
        
        Returns:
            Dictionary with proxy configuration or None if no proxies available
        """
        # If no proxies or all failed, return None
        available_proxies = [p for p in self.proxies if p not in self.failed_proxies]
        
        if not available_proxies:
            return None
        
        # Choose a random proxy
        proxy = random.choice(available_proxies)
        
        # Avoid using the same proxy consecutively if possible
        if proxy == self.last_proxy and len(available_proxies) > 1:
            available_proxies.remove(proxy)
            proxy = random.choice(available_proxies)
        
        self.last_proxy = proxy
        
        return {
            'http': proxy,
            'https': proxy
        }
    
    def mark_proxy_failed(self, proxy_url):
        """
        Mark a proxy as failed
        
        Args:
            proxy_url: URL of the proxy that failed
        """
        if proxy_url in self.proxies:
            self.failed_proxies.add(proxy_url)
            logger.warning(f"Marked proxy as failed: {proxy_url}")
    
    def reset_failed_proxies(self):
        """Reset the list of failed proxies"""
        self.failed_proxies = set()
        logger.info("Reset failed proxy list")
    
    def has_proxies(self):
        """Check if any proxies are configured"""
        return len(self.proxies) > 0