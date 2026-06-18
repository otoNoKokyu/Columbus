import asyncio
import time

class RateLimiter:
    """
    Ensures strict rate limiting:
    - Maximum N concurrent requests (e.g., 2).
    - Maximum M requests per time window (e.g., 10 per 60 seconds).
    """
    def __init__(self, max_concurrent: int, max_requests_per_window: int, window_seconds: float):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_requests = max_requests_per_window
        self.window_seconds = window_seconds
        
        # Keep track of timestamps when requests were made
        self.request_timestamps = []

    async def acquire(self):
        """
        Wait until we are allowed to make a new request based on both
        concurrency limits and time-window limits.
        """
        # First, enforce time window limit (token bucket / rolling window)
        while True:
            now = time.time()
            
            # Clean up old timestamps outside the current window
            self.request_timestamps = [
                ts for ts in self.request_timestamps 
                if now - ts < self.window_seconds
            ]
            
            # Check if we have capacity in the rolling window
            if len(self.request_timestamps) < self.max_requests:
                break
                
            # If not, wait for the oldest request to age out of the window
            oldest_ts = self.request_timestamps[0]
            sleep_time = self.window_seconds - (now - oldest_ts)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        # Record this request timestamp
        self.request_timestamps.append(time.time())
        
        # Then, wait for a concurrency slot
        await self.semaphore.acquire()

    def release(self):
        """
        Release the concurrency slot. 
        Note: The time-window record is NOT removed here, as it must persist for the window duration.
        """
        self.semaphore.release()
