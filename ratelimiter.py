import time
from typing import List, Tuple
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit=8, window=60, exempt_routes: List[Tuple[str, str]] = []):
        super().__init__(app)
        self.limit = limit  # Number of requests allowed
        self.window = window  # Time window in seconds
        self.ip_cache = {}
        self.exempt_routes = set(exempt_routes)

    async def dispatch(self, request: Request, call_next):
        # Check if the current route is exempt
        current_route = (request.url.path, request.method)
        if current_route in self.exempt_routes:
            return await call_next(request)

        ip = request.client.host
        current_time = time.time()

        if ip in self.ip_cache:
            last_reset, count = self.ip_cache[ip]
            if current_time - last_reset > self.window:
                # Reset if the window has passed
                self.ip_cache[ip] = (current_time, 1)
            elif count >= self.limit:
                # Return a JSONResponse for rate limit exceeded
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded. Please try again later."},
                )
            else:
                self.ip_cache[ip] = (last_reset, count + 1)
        else:
            self.ip_cache[ip] = (current_time, 1)

        response = await call_next(request)
        return response
