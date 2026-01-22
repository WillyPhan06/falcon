from __future__ import annotations

import asyncio
import threading
import time
import urllib.parse
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import TYPE_CHECKING

from ._typing import UniversalMiddlewareWithProcessResponse
from .errors import HTTPRequestProcessingTimeout
from .util import code_to_http_status

if TYPE_CHECKING:
    from .asgi.request import Request as AsgiRequest
    from .asgi.response import Response as AsgiResponse
    from .request import Request
    from .response import Response


DEFAULT_IDEMPOTENCY_HEADER = 'Idempotency-Key'
DEFAULT_IDEMPOTENCY_TTL = 86400  # 24 hours in seconds

# Sentinel value indicating a request is currently being processed
_PROCESSING = object()


@dataclass
class CachedResponse:
    """Cached response data for idempotent requests.

    Attributes:
        status: The HTTP status string (e.g., '200 OK', '201 Created').
        headers: A dictionary of response headers.
        media: The serializable media object (if set).
        data: The raw byte response data (if set).
        text: The text response content (if set).
        timestamp: Unix timestamp when this response was cached.
    """

    status: str
    headers: dict[str, str]
    media: Any
    data: bytes | None
    text: str | None
    timestamp: float


class IdempotencyStore(Protocol):
    """Protocol for idempotency stores.

    Implementations of this protocol are used to store and retrieve cached
    responses for idempotent requests. The store is responsible for managing
    the lifecycle of cached entries, including expiration.

    Example:
        A simple Redis-based implementation might look like::

            import json
            import redis

            class RedisIdempotencyStore:
                def __init__(self, redis_client, ttl=86400):
                    self.redis = redis_client
                    self.ttl = ttl

                def get(self, key):
                    data = self.redis.get(f'idempotency:{key}')
                    if data:
                        d = json.loads(data)
                        return CachedResponse(**d)
                    return None

                def set(self, key, response, ttl=None):
                    ttl = ttl or self.ttl
                    data = json.dumps({
                        'status': response.status,
                        'headers': response.headers,
                        'media': response.media,
                        'data': response.data.decode() if response.data else None,
                        'text': response.text,
                        'timestamp': response.timestamp,
                    })
                    self.redis.setex(f'idempotency:{key}', ttl, data)

                async def get_async(self, key):
                    return self.get(key)

                async def set_async(self, key, response, ttl=None):
                    self.set(key, response, ttl)
    """

    def get(self, key: str) -> CachedResponse | None:
        """Retrieve a cached response by idempotency key.

        Args:
            key: The idempotency key to look up.

        Returns:
            The cached response if found and not expired, otherwise None.
        """
        ...

    def set(self, key: str, response: CachedResponse, ttl: int | None = None) -> None:
        """Store a cached response.

        Args:
            key: The idempotency key to store under.
            response: The cached response data.
            ttl: Optional time-to-live in seconds. If not provided,
                the store's default TTL should be used.
        """
        ...

    async def get_async(self, key: str) -> CachedResponse | None:
        """Async version of get().

        Args:
            key: The idempotency key to look up.

        Returns:
            The cached response if found and not expired, otherwise None.
        """
        ...

    async def set_async(
        self, key: str, response: CachedResponse, ttl: int | None = None
    ) -> None:
        """Async version of set().

        Args:
            key: The idempotency key to store under.
            response: The cached response data.
            ttl: Optional time-to-live in seconds. If not provided,
                the store's default TTL should be used.
        """
        ...


class InMemoryIdempotencyStore:
    """In-memory implementation of IdempotencyStore.

    This store keeps cached responses in memory using a dictionary. It is
    suitable for development, testing, and single-process deployments.

    This implementation includes built-in protection against race conditions
    when multiple requests with the same idempotency key arrive concurrently.
    Only the first request will be processed; subsequent requests will wait
    for the first to complete and then receive the cached response.

    Warning:
        This store does not share state between multiple processes. For
        multi-process deployments (e.g., behind a load balancer), consider
        using a distributed store such as Redis with proper locking.

    Warning:
        This store has no maximum size limit. For production use with high
        traffic, consider implementing a store with LRU eviction or using
        an external cache.

    Keyword Arguments:
        ttl (int): Default time-to-live for cached entries in seconds.
            Defaults to 86400 (24 hours).

    Example:
        Create and use an in-memory store::

            store = InMemoryIdempotencyStore(ttl=3600)  # 1 hour TTL
            app = falcon.App(middleware=[
                IdempotencyMiddleware(store=store)
            ])
    """

    def __init__(self, ttl: int = DEFAULT_IDEMPOTENCY_TTL) -> None:
        self._cache: dict[str, CachedResponse | object] = {}
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Event] = {}
        self._async_key_locks: dict[str, asyncio.Event] = {}
        self.ttl = ttl

    def _is_expired(self, response: CachedResponse, ttl: int | None = None) -> bool:
        """Check if a cached response has expired."""
        ttl = ttl if ttl is not None else self.ttl
        return time.time() - response.timestamp > ttl

    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        now = time.time()
        expired_keys = [
            key
            for key, resp in self._cache.items()
            if isinstance(resp, CachedResponse) and now - resp.timestamp > self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]

    def get(self, key: str) -> CachedResponse | None:
        """Retrieve a cached response by idempotency key.

        Args:
            key: The idempotency key to look up.

        Returns:
            The cached response if found and not expired, otherwise None.
        """
        with self._lock:
            response = self._cache.get(key)
            if response is None:
                return None
            if isinstance(response, CachedResponse):
                if self._is_expired(response):
                    del self._cache[key]
                    return None
                return response
            # response is _PROCESSING sentinel - should not happen in normal get()
            return None

    def set(self, key: str, response: CachedResponse, ttl: int | None = None) -> None:
        """Store a cached response.

        Args:
            key: The idempotency key to store under.
            response: The cached response data.
            ttl: Optional time-to-live in seconds (not used for storage,
                entries use the store's default TTL for expiration checks).
        """
        with self._lock:
            # Periodic cleanup to prevent unbounded growth
            if len(self._cache) > 1000:
                self._cleanup_expired()
            self._cache[key] = response
            # Signal any waiting threads that the response is ready
            if key in self._key_locks:
                self._key_locks[key].set()

    def acquire_or_wait(self, key: str) -> CachedResponse | None:
        """Try to acquire processing rights for a key, or wait if already processing.

        This method handles race conditions by ensuring only one request processes
        a given idempotency key at a time. If another request is already processing
        this key, this method will block until the response is available.

        Args:
            key: The idempotency key.

        Returns:
            CachedResponse if the key was already cached or became cached while waiting.
            None if this caller should process the request (acquired the lock).
        """
        with self._lock:
            cached = self._cache.get(key)

            # If we have a cached response, return it
            if isinstance(cached, CachedResponse):
                if self._is_expired(cached):
                    del self._cache[key]
                else:
                    return cached

            # If another request is processing this key, wait for it
            if cached is _PROCESSING:
                event = self._key_locks.get(key)
                if event is None:
                    event = threading.Event()
                    self._key_locks[key] = event
            else:
                # Mark this key as being processed
                self._cache[key] = _PROCESSING
                self._key_locks[key] = threading.Event()
                return None

        # Wait outside the lock for the processing to complete
        event.wait(timeout=30.0)  # 30 second timeout to avoid infinite waits

        # After waiting, get the cached response
        with self._lock:
            cached = self._cache.get(key)
            if isinstance(cached, CachedResponse) and not self._is_expired(cached):
                return cached
            return None

    def release(self, key: str) -> None:
        """Release processing lock for a key (called on failure/error).

        Args:
            key: The idempotency key.
        """
        with self._lock:
            if self._cache.get(key) is _PROCESSING:
                del self._cache[key]
            if key in self._key_locks:
                self._key_locks[key].set()
                del self._key_locks[key]

    async def get_async(self, key: str) -> CachedResponse | None:
        """Async version of get().

        Note:
            This implementation simply calls the synchronous version since
            in-memory operations are fast enough not to block the event loop.
        """
        return self.get(key)

    async def set_async(
        self, key: str, response: CachedResponse, ttl: int | None = None
    ) -> None:
        """Async version of set().

        Note:
            This implementation simply calls the synchronous version since
            in-memory operations are fast enough not to block the event loop.
        """
        with self._lock:
            if len(self._cache) > 1000:
                self._cleanup_expired()
            self._cache[key] = response
            # Signal any waiting coroutines
            if key in self._async_key_locks:
                self._async_key_locks[key].set()
            if key in self._key_locks:
                self._key_locks[key].set()

    async def acquire_or_wait_async(self, key: str) -> CachedResponse | None:
        """Async version of acquire_or_wait().

        Args:
            key: The idempotency key.

        Returns:
            CachedResponse if the key was already cached or became cached while waiting.
            None if this caller should process the request (acquired the lock).
        """
        with self._lock:
            cached = self._cache.get(key)

            # If we have a cached response, return it
            if isinstance(cached, CachedResponse):
                if self._is_expired(cached):
                    del self._cache[key]
                else:
                    return cached

            # If another request is processing this key, wait for it
            if cached is _PROCESSING:
                if key not in self._async_key_locks:
                    self._async_key_locks[key] = asyncio.Event()
                event = self._async_key_locks[key]
            else:
                # Mark this key as being processed
                self._cache[key] = _PROCESSING
                self._async_key_locks[key] = asyncio.Event()
                return None

        # Wait outside the lock for the processing to complete
        try:
            await asyncio.wait_for(event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            pass

        # After waiting, get the cached response
        with self._lock:
            cached = self._cache.get(key)
            if isinstance(cached, CachedResponse) and not self._is_expired(cached):
                return cached
            return None

    async def release_async(self, key: str) -> None:
        """Async version of release().

        Args:
            key: The idempotency key.
        """
        with self._lock:
            if self._cache.get(key) is _PROCESSING:
                del self._cache[key]
            if key in self._async_key_locks:
                self._async_key_locks[key].set()
                del self._async_key_locks[key]
            if key in self._key_locks:
                self._key_locks[key].set()
                del self._key_locks[key]


class IdempotencyMiddleware:
    """Idempotency Middleware.

    This middleware allows clients to safely retry requests without causing
    duplicate side effects (such as duplicate database inserts or state changes).
    When a request includes an idempotency key header, the middleware caches
    the response and returns the cached response for subsequent requests
    with the same key.

    This is particularly useful for:

    * Payment processing where duplicate charges must be avoided
    * Resource creation where duplicate inserts would cause errors
    * Any operation where network failures might cause clients to retry

    The idempotency key is typically provided by the client in a request header
    (default: ``Idempotency-Key``). If no key is provided, the request is
    processed normally without caching.

    Keyword Arguments:
        store (IdempotencyStore): The storage backend for cached responses.
            Defaults to an :class:`InMemoryIdempotencyStore` instance.
        header_name (str): The name of the header containing the idempotency key.
            Defaults to ``'Idempotency-Key'``.
        ttl (int): Time-to-live for cached responses in seconds.
            Defaults to 86400 (24 hours).
        methods (Iterable[str]): HTTP methods to apply idempotency to.
            Defaults to ``('POST', 'PUT', 'PATCH')``. GET, DELETE, HEAD, and
            OPTIONS are naturally idempotent and typically don't need this.

    Example:
        Basic usage with default in-memory store::

            import falcon

            app = falcon.App(middleware=[
                falcon.IdempotencyMiddleware()
            ])

        With custom configuration::

            middleware = falcon.IdempotencyMiddleware(
                header_name='X-Request-Id',
                ttl=3600,  # 1 hour
                methods=['POST'],  # Only POST requests
            )
            app = falcon.App(middleware=[middleware])

        With a custom store (e.g., Redis)::

            redis_store = MyRedisIdempotencyStore(redis_client)
            app = falcon.App(middleware=[
                falcon.IdempotencyMiddleware(store=redis_store)
            ])

    Note:
        The cached response includes status, headers, and body content
        (media, data, or text). The original response is replayed exactly
        when a duplicate request is received.

    Note:
        For ASGI applications, this middleware uses async storage operations
        via ``get_async`` and ``set_async`` methods on the store.
    """

    def __init__(
        self,
        store: IdempotencyStore | None = None,
        header_name: str = DEFAULT_IDEMPOTENCY_HEADER,
        ttl: int = DEFAULT_IDEMPOTENCY_TTL,
        methods: Iterable[str] = ('POST', 'PUT', 'PATCH'),
    ) -> None:
        self._store = store if store is not None else InMemoryIdempotencyStore(ttl=ttl)
        self._header_name = header_name
        self._ttl = ttl
        self._methods = frozenset(m.upper() for m in methods)

    def _get_idempotency_key(self, req: Request | AsgiRequest) -> str | None:
        """Extract the idempotency key from the request."""
        return req.get_header(self._header_name)

    def _create_cache_key(
        self, req: Request | AsgiRequest, idempotency_key: str
    ) -> str:
        """Create a unique cache key from request attributes.

        The cache key combines the idempotency key with the request method,
        path, and query string to ensure uniqueness across different endpoints
        and query parameters.
        """
        query_string = req.query_string
        if query_string:
            return f'{req.method}:{req.path}?{query_string}:{idempotency_key}'
        return f'{req.method}:{req.path}:{idempotency_key}'

    def _restore_response(
        self, resp: Response | AsgiResponse, cached: CachedResponse
    ) -> None:
        """Restore a cached response to the response object."""
        resp.status = cached.status
        for name, value in cached.headers.items():
            resp.set_header(name, value)
        if cached.media is not None:
            resp.media = cached.media
        elif cached.data is not None:
            resp.data = cached.data
        elif cached.text is not None:
            resp.text = cached.text

    def _normalize_status(self, status: str | int) -> str:
        """Normalize status to string format.

        Args:
            status: The status code or string.

        Returns:
            A status string like '200 OK' or '201 Created'.
        """
        if isinstance(status, int):
            return code_to_http_status(status)
        return str(status)

    def _capture_response(self, resp: Response | AsgiResponse) -> CachedResponse:
        """Capture the current response state for caching."""
        return CachedResponse(
            status=self._normalize_status(resp.status),
            headers=resp.headers,
            media=resp._media,
            data=resp._data,
            text=resp.text,
            timestamp=time.time(),
        )

    def _has_locking_support(self, store: IdempotencyStore) -> bool:
        """Check if the store supports locking methods."""
        return hasattr(store, 'acquire_or_wait') and hasattr(store, 'release')

    def _has_async_locking_support(self, store: IdempotencyStore) -> bool:
        """Check if the store supports async locking methods."""
        return hasattr(store, 'acquire_or_wait_async') and hasattr(
            store, 'release_async'
        )

    def process_request(self, req: Request, resp: Response) -> None:
        """Process the request before routing (WSGI).

        If an idempotency key is present and a cached response exists,
        restore the cached response and mark the request as complete.

        This method handles race conditions by using locking when supported
        by the store, ensuring only one request processes a given key.
        """
        if req.method not in self._methods:
            return

        idempotency_key = self._get_idempotency_key(req)
        if idempotency_key is None:
            return

        cache_key = self._create_cache_key(req, idempotency_key)

        # Use locking if supported by the store
        if self._has_locking_support(self._store):
            cached = self._store.acquire_or_wait(cache_key)
            if cached is not None:
                self._restore_response(resp, cached)
                resp.complete = True
                return
            # Mark that we need to cache or release on response
            req.context._idempotency_cache_key = cache_key
            req.context._idempotency_acquired_lock = True
        else:
            # Fallback for stores without locking support
            cached = self._store.get(cache_key)
            if cached is not None:
                self._restore_response(resp, cached)
                resp.complete = True
                return
            req.context._idempotency_cache_key = cache_key
            req.context._idempotency_acquired_lock = False

    def process_response(
        self, req: Request, resp: Response, resource: object, req_succeeded: bool
    ) -> None:
        """Process the response after routing (WSGI).

        If the request had an idempotency key and succeeded, cache the response.
        If the request failed and we acquired a lock, release it.
        """
        cache_key = getattr(req.context, '_idempotency_cache_key', None)
        if cache_key is None:
            return

        acquired_lock = getattr(req.context, '_idempotency_acquired_lock', False)

        # Only cache successful responses
        if not req_succeeded:
            # Release the lock if we acquired one and the request failed
            if acquired_lock:
                self._store.release(cache_key)
            return

        cached_response = self._capture_response(resp)
        self._store.set(cache_key, cached_response, self._ttl)

    async def process_request_async(
        self, req: AsgiRequest, resp: AsgiResponse
    ) -> None:
        """Process the request before routing (ASGI).

        If an idempotency key is present and a cached response exists,
        restore the cached response and mark the request as complete.

        This method handles race conditions by using async locking when
        supported by the store, ensuring only one request processes a given key.
        """
        if req.method not in self._methods:
            return

        idempotency_key = self._get_idempotency_key(req)
        if idempotency_key is None:
            return

        cache_key = self._create_cache_key(req, idempotency_key)

        # Use async locking if supported by the store
        if self._has_async_locking_support(self._store):
            cached = await self._store.acquire_or_wait_async(cache_key)
            if cached is not None:
                self._restore_response(resp, cached)
                resp.complete = True
                return
            # Mark that we need to cache or release on response
            req.context._idempotency_cache_key = cache_key
            req.context._idempotency_acquired_lock = True
        else:
            # Fallback for stores without async locking support
            cached = await self._store.get_async(cache_key)
            if cached is not None:
                self._restore_response(resp, cached)
                resp.complete = True
                return
            req.context._idempotency_cache_key = cache_key
            req.context._idempotency_acquired_lock = False

    async def process_response_async(
        self,
        req: AsgiRequest,
        resp: AsgiResponse,
        resource: object,
        req_succeeded: bool,
    ) -> None:
        """Process the response after routing (ASGI).

        If the request had an idempotency key and succeeded, cache the response.
        If the request failed and we acquired a lock, release it.
        """
        cache_key = getattr(req.context, '_idempotency_cache_key', None)
        if cache_key is None:
            return

        acquired_lock = getattr(req.context, '_idempotency_acquired_lock', False)

        # Only cache successful responses
        if not req_succeeded:
            # Release the lock if we acquired one and the request failed
            if acquired_lock:
                await self._store.release_async(cache_key)
            return

        cached_response = self._capture_response(resp)
        await self._store.set_async(cache_key, cached_response, self._ttl)


class CORSMiddleware(UniversalMiddlewareWithProcessResponse):
    """CORS Middleware.

    This middleware provides a simple out-of-the box CORS policy, including handling
    of preflighted requests from the browser.

    See also:

    * https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
    * https://www.w3.org/TR/cors/#resource-processing-model

    Note:
        Falcon will automatically add OPTIONS responders if they are missing from the
        responder instances added to the routes. When providing a custom ``on_options``
        method, the ``Allow`` headers in the response should be set to the allowed
        method values. If the ``Allow`` header is missing from the response,
        this middleware will deny the preflight request.

        This is also valid when using a sink function.

    Keyword Arguments:
        allow_origins (Union[str, Iterable[str]]): List of origins to allow (case
            sensitive). The string ``'*'`` acts as a wildcard, matching every origin.
            (default ``'*'``).
        expose_headers (Optional[Union[str, Iterable[str]]]): List of additional
            response headers to expose via the ``Access-Control-Expose-Headers``
            header. These headers are in addition to the CORS-safelisted ones:
            ``Cache-Control``, ``Content-Language``, ``Content-Length``,
            ``Content-Type``, ``Expires``, ``Last-Modified``, ``Pragma``.
            (default ``None``).

            See also:
            https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Expose-Headers
        allow_credentials (Optional[Union[str, Iterable[str]]]): List of origins
            (case sensitive) for which to allow credentials via the
            ``Access-Control-Allow-Credentials`` header.
            The string ``'*'`` acts as a wildcard, matching every allowed origin,
            while ``None`` disallows all origins. This parameter takes effect only
            if the origin is allowed by the ``allow_origins`` argument.
            (default ``None``).
        allow_private_network (bool):
            If ``True``, the server includes the
            ``Access-Control-Allow-Private-Network`` header in responses to
            CORS preflight (OPTIONS) requests. This indicates that the resource is
            willing to respond to requests from less-public IP address spaces
            (e.g., from public site to private device).
            (default ``False``).

            See also:
            https://wicg.github.io/private-network-access/#private-network-request-heading
    """

    def __init__(
        self,
        allow_origins: str | Iterable[str] = '*',
        expose_headers: str | Iterable[str] | None = None,
        allow_credentials: str | Iterable[str] | None = None,
        allow_private_network: bool = False,
    ):
        if allow_origins == '*':
            self.allow_origins = allow_origins
        else:
            if isinstance(allow_origins, str):
                allow_origins = [allow_origins]
            self.allow_origins = frozenset(allow_origins)
            if '*' in self.allow_origins:
                raise ValueError(
                    'The wildcard string "*" may only be passed to allow_origins as a '
                    'string literal, not inside an iterable.'
                )

        if expose_headers is not None and not isinstance(expose_headers, str):
            expose_headers = ', '.join(expose_headers)
        self.expose_headers = expose_headers

        if allow_credentials is None:
            allow_credentials = frozenset()
        elif allow_credentials != '*':
            if isinstance(allow_credentials, str):
                allow_credentials = [allow_credentials]
            allow_credentials = frozenset(allow_credentials)
            if '*' in allow_credentials:
                raise ValueError(
                    'The wildcard string "*" may only be passed to allow_credentials '
                    'as a string literal, not inside an iterable.'
                )
        self.allow_credentials = allow_credentials
        self.allow_private_network = allow_private_network

    def process_response(
        self, req: Request, resp: Response, resource: object, req_succeeded: bool
    ) -> None:
        """Implement the CORS policy for all routes.

        This middleware provides a simple out-of-the box CORS policy,
        including handling of preflighted requests from the browser.

        See also: https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS

        See also: https://www.w3.org/TR/cors/#resource-processing-model
        """

        origin = req.get_header('Origin')
        if origin is None:
            return

        if self.allow_origins != '*' and origin not in self.allow_origins:
            return

        if resp.get_header('Access-Control-Allow-Origin') is None:
            set_origin = '*' if self.allow_origins == '*' else origin
            if self.allow_credentials == '*' or origin in self.allow_credentials:
                set_origin = origin
                resp.set_header('Access-Control-Allow-Credentials', 'true')
            resp.set_header('Access-Control-Allow-Origin', set_origin)

        if self.expose_headers:
            resp.set_header('Access-Control-Expose-Headers', self.expose_headers)

        if (
            req_succeeded
            and req.method == 'OPTIONS'
            and req.get_header('Access-Control-Request-Method')
        ):
            # NOTE(kgriffs): This is a CORS preflight request. Patch the
            #   response accordingly.

            allow = resp.get_header('Allow')
            resp.delete_header('Allow')

            allow_headers = req.get_header(
                'Access-Control-Request-Headers', default='*'
            )

            if allow is None:
                # there is no allow set, remove all access control headers
                resp.delete_header('Access-Control-Allow-Methods')
                resp.delete_header('Access-Control-Allow-Headers')
                resp.delete_header('Access-Control-Max-Age')
                resp.delete_header('Access-Control-Expose-Headers')
                resp.delete_header('Access-Control-Allow-Origin')
            else:
                resp.set_header('Access-Control-Allow-Methods', allow)
                resp.set_header('Access-Control-Allow-Headers', allow_headers)
                resp.set_header('Access-Control-Max-Age', '86400')  # 24 hours

            if self.allow_private_network and (
                req.get_header('Access-Control-Request-Private-Network') == 'true'
            ):
                resp.set_header('Access-Control-Allow-Private-Network', 'true')

    async def process_response_async(
        self,
        req: AsgiRequest,
        resp: AsgiResponse,
        resource: object,
        req_succeeded: bool,
    ) -> None:
        self.process_response(req, resp, resource, req_succeeded)


DEFAULT_REQUEST_ID_HEADER = 'X-Request-ID'


class RequestIDMiddleware:
    """Request ID Middleware.

    This middleware automatically extracts a request ID from the incoming request
    or generates one if not present. The request ID is stored in the request
    context and added to the response headers for tracing and correlation purposes.

    The request ID can be accessed in responders via ``req.context.request_id``
    or via the ``req.request_id`` property (which reads the header value).

    This is useful for:

    * Distributed tracing across microservices
    * Log correlation and debugging
    * Request tracking through API gateways and load balancers

    Keyword Arguments:
        header_name (str): The name of the header to read/write the request ID.
            Defaults to ``'X-Request-ID'``.
        generator (callable): A callable that returns a new unique request ID.
            If not specified, defaults to generating a UUID4 string.

    Example:
        Basic usage with default UUID generator::

            import falcon

            app = falcon.App(middleware=[
                falcon.RequestIDMiddleware()
            ])

        With custom configuration::

            import uuid

            # Use a custom generator
            def custom_generator():
                return f'req-{uuid.uuid4().hex[:8]}'

            middleware = falcon.RequestIDMiddleware(
                header_name='X-Correlation-ID',
                generator=custom_generator,
            )
            app = falcon.App(middleware=[middleware])

        Accessing the request ID in a responder::

            class MyResource:
                def on_get(self, req, resp):
                    # Access from context (works even if middleware generated it)
                    request_id = req.context.request_id

                    # Or from the header property (reads original header)
                    request_id = req.request_id

                    resp.media = {'request_id': request_id}

    Note:
        The request ID is always added to the response headers, regardless
        of whether the request succeeded or failed.

    Note:
        For ASGI applications, this middleware uses async methods that
        simply call their sync counterparts, as the operations are lightweight.

    .. versionadded:: 4.1
    """

    def __init__(
        self,
        header_name: str = DEFAULT_REQUEST_ID_HEADER,
        generator: Any = None,
    ) -> None:
        self._header_name = header_name
        if generator is None:
            import uuid

            self._generator = lambda: str(uuid.uuid4())
        else:
            self._generator = generator

    def _get_request_id(self, req: Request | AsgiRequest) -> str:
        """Extract request ID from header or generate a new one."""
        request_id = req.get_header(self._header_name)
        if not request_id:
            request_id = self._generator()
        return request_id

    def _apply_request_id(
        self, req: Request | AsgiRequest, resp: Response | AsgiResponse
    ) -> None:
        """Apply request ID to the request context and response header.

        This is the shared logic used by both WSGI and ASGI request processing.
        It extracts the request ID from the incoming header or generates a new one,
        stores it in the request context, and sets it on the response header.
        """
        request_id = self._get_request_id(req)
        req.context.request_id = request_id
        resp.set_header(self._header_name, request_id)

    def _ensure_response_header(
        self, req: Request | AsgiRequest, resp: Response | AsgiResponse
    ) -> None:
        """Ensure the request ID is present in the response headers.

        This is the shared logic used by both WSGI and ASGI response processing.
        It ensures the request ID is present in the response even if the request
        failed before ``process_request`` completed.
        """
        request_id = getattr(req.context, 'request_id', None)
        if request_id and not resp.get_header(self._header_name):
            resp.set_header(self._header_name, request_id)

    def process_request(self, req: Request, resp: Response) -> None:
        """Process the request and assign a request ID (WSGI).

        Extracts the request ID from the incoming header or generates
        a new one if not present. Stores it in ``req.context.request_id``
        and sets it on the response header.
        """
        self._apply_request_id(req, resp)

    def process_response(
        self, req: Request, resp: Response, resource: object, req_succeeded: bool
    ) -> None:
        """Ensure the request ID is in the response headers (WSGI).

        This method ensures the request ID is present in the response even if
        the request failed before ``process_request`` completed.
        """
        self._ensure_response_header(req, resp)

    async def process_request_async(
        self, req: AsgiRequest, resp: AsgiResponse
    ) -> None:
        """Process the request and assign a request ID (ASGI)."""
        self._apply_request_id(req, resp)

    async def process_response_async(
        self,
        req: AsgiRequest,
        resp: AsgiResponse,
        resource: object,
        req_succeeded: bool,
    ) -> None:
        """Ensure the request ID is in the response headers (ASGI)."""
        self._ensure_response_header(req, resp)


DEFAULT_TIMEOUT = 30.0

DEFAULT_CACHE_TTL = 300  # 5 minutes in seconds
DEFAULT_CACHE_METHODS = frozenset(['GET'])
DEFAULT_INVALIDATION_METHODS = frozenset(['POST', 'PUT', 'PATCH', 'DELETE'])


class CacheStore(Protocol):
    """Protocol for cache stores.

    Implementations of this protocol are used to store and retrieve cached
    responses for GET requests. The store is responsible for managing
    the lifecycle of cached entries, including expiration.

    Example:
        A simple Redis-based implementation might look like::

            import json
            import redis

            class RedisCacheStore:
                def __init__(self, redis_client, ttl=300):
                    self.redis = redis_client
                    self.ttl = ttl

                def get(self, key):
                    data = self.redis.get(f'cache:{key}')
                    if data:
                        d = json.loads(data)
                        return CachedResponse(**d)
                    return None

                def set(self, key, response, ttl=None):
                    ttl = ttl or self.ttl
                    data = json.dumps({
                        'status': response.status,
                        'headers': response.headers,
                        'media': response.media,
                        'data': response.data.decode() if response.data else None,
                        'text': response.text,
                        'timestamp': response.timestamp,
                    })
                    self.redis.setex(f'cache:{key}', ttl, data)

                def invalidate(self, key):
                    self.redis.delete(f'cache:{key}')

                def invalidate_prefix(self, prefix):
                    keys = self.redis.keys(f'cache:{prefix}*')
                    if keys:
                        self.redis.delete(*keys)

                async def get_async(self, key):
                    return self.get(key)

                async def set_async(self, key, response, ttl=None):
                    self.set(key, response, ttl)

                async def invalidate_async(self, key):
                    self.invalidate(key)

                async def invalidate_prefix_async(self, prefix):
                    self.invalidate_prefix(prefix)
    """

    def get(self, key: str) -> CachedResponse | None:
        """Retrieve a cached response by cache key.

        Args:
            key: The cache key to look up.

        Returns:
            The cached response if found and not expired, otherwise None.
        """
        ...

    def set(self, key: str, response: CachedResponse, ttl: int | None = None) -> None:
        """Store a cached response.

        Args:
            key: The cache key to store under.
            response: The cached response data.
            ttl: Optional time-to-live in seconds. If not provided,
                the store's default TTL should be used.
        """
        ...

    def invalidate(self, key: str) -> None:
        """Invalidate (remove) a specific cached entry.

        Args:
            key: The cache key to invalidate.
        """
        ...

    def invalidate_prefix(self, prefix: str) -> None:
        """Invalidate all cached entries matching a prefix.

        Args:
            prefix: The prefix to match. All keys starting with this
                prefix should be invalidated.
        """
        ...

    async def get_async(self, key: str) -> CachedResponse | None:
        """Async version of get().

        Args:
            key: The cache key to look up.

        Returns:
            The cached response if found and not expired, otherwise None.
        """
        ...

    async def set_async(
        self, key: str, response: CachedResponse, ttl: int | None = None
    ) -> None:
        """Async version of set().

        Args:
            key: The cache key to store under.
            response: The cached response data.
            ttl: Optional time-to-live in seconds. If not provided,
                the store's default TTL should be used.
        """
        ...

    async def invalidate_async(self, key: str) -> None:
        """Async version of invalidate().

        Args:
            key: The cache key to invalidate.
        """
        ...

    async def invalidate_prefix_async(self, prefix: str) -> None:
        """Async version of invalidate_prefix().

        Args:
            prefix: The prefix to match. All keys starting with this
                prefix should be invalidated.
        """
        ...


class InMemoryCacheStore:
    """In-memory implementation of CacheStore.

    This store keeps cached responses in memory using a dictionary. It is
    suitable for development, testing, and single-process deployments.

    Warning:
        This store does not share state between multiple processes. For
        multi-process deployments (e.g., behind a load balancer), consider
        using a distributed store such as Redis.

    Warning:
        This store has no maximum size limit. For production use with high
        traffic, consider implementing a store with LRU eviction or using
        an external cache.

    Keyword Arguments:
        ttl (int): Default time-to-live for cached entries in seconds.
            Defaults to 300 (5 minutes).

    Example:
        Create and use an in-memory store::

            store = InMemoryCacheStore(ttl=60)  # 1 minute TTL
            app = falcon.App(middleware=[
                CacheMiddleware(store=store)
            ])
    """

    def __init__(self, ttl: int = DEFAULT_CACHE_TTL) -> None:
        self._cache: dict[str, CachedResponse] = {}
        self._lock = threading.Lock()
        self.ttl = ttl

    def _is_expired(self, response: CachedResponse, ttl: int | None = None) -> bool:
        """Check if a cached response has expired."""
        ttl = ttl if ttl is not None else self.ttl
        return time.time() - response.timestamp > ttl

    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        now = time.time()
        expired_keys = [
            key
            for key, resp in self._cache.items()
            if now - resp.timestamp > self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]

    def get(self, key: str) -> CachedResponse | None:
        """Retrieve a cached response by cache key.

        Args:
            key: The cache key to look up.

        Returns:
            The cached response if found and not expired, otherwise None.
        """
        with self._lock:
            response = self._cache.get(key)
            if response is None:
                return None
            if self._is_expired(response):
                del self._cache[key]
                return None
            return response

    def set(self, key: str, response: CachedResponse, ttl: int | None = None) -> None:
        """Store a cached response.

        Args:
            key: The cache key to store under.
            response: The cached response data.
            ttl: Optional time-to-live in seconds (not used for storage,
                entries use the store's default TTL for expiration checks).
        """
        with self._lock:
            # Periodic cleanup to prevent unbounded growth
            if len(self._cache) > 1000:
                self._cleanup_expired()
            self._cache[key] = response

    def invalidate(self, key: str) -> None:
        """Invalidate (remove) a specific cached entry.

        Args:
            key: The cache key to invalidate.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def invalidate_prefix(self, prefix: str) -> None:
        """Invalidate all cached entries matching a prefix.

        Args:
            prefix: The prefix to match. All keys starting with this
                prefix should be invalidated.
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys() if key.startswith(prefix)
            ]
            for key in keys_to_remove:
                del self._cache[key]

    async def get_async(self, key: str) -> CachedResponse | None:
        """Async version of get().

        Note:
            This implementation simply calls the synchronous version since
            in-memory operations are fast enough not to block the event loop.
        """
        return self.get(key)

    async def set_async(
        self, key: str, response: CachedResponse, ttl: int | None = None
    ) -> None:
        """Async version of set().

        Note:
            This implementation simply calls the synchronous version since
            in-memory operations are fast enough not to block the event loop.
        """
        self.set(key, response, ttl)

    async def invalidate_async(self, key: str) -> None:
        """Async version of invalidate().

        Note:
            This implementation simply calls the synchronous version since
            in-memory operations are fast enough not to block the event loop.
        """
        self.invalidate(key)

    async def invalidate_prefix_async(self, prefix: str) -> None:
        """Async version of invalidate_prefix().

        Note:
            This implementation simply calls the synchronous version since
            in-memory operations are fast enough not to block the event loop.
        """
        self.invalidate_prefix(prefix)


class CacheMiddleware:
    """Response Caching Middleware.

    This middleware caches responses to GET requests based on the request URI
    and query string, allowing subsequent identical requests to be served from
    cache without re-executing the resource handler.

    Cache entries are automatically invalidated when non-GET requests (POST,
    PUT, PATCH, DELETE) are made to the same path, ensuring data consistency.

    This is useful for:

    * Reducing server load for frequently accessed read-only endpoints
    * Improving response times for expensive GET operations
    * Caching API responses that don't change frequently

    Keyword Arguments:
        store (CacheStore): The storage backend for cached responses.
            Defaults to an :class:`InMemoryCacheStore` instance.
        ttl (int): Time-to-live for cached responses in seconds.
            Defaults to 300 (5 minutes).
        methods (Iterable[str]): HTTP methods to cache responses for.
            Defaults to ``('GET',)``. Only GET requests are cacheable by
            default since other methods typically have side effects.
        invalidation_methods (Iterable[str]): HTTP methods that should
            invalidate cache entries for the same path. Defaults to
            ``('POST', 'PUT', 'PATCH', 'DELETE')``.

    Example:
        Basic usage with default in-memory store::

            import falcon

            app = falcon.App(middleware=[
                falcon.CacheMiddleware()
            ])

        With custom configuration::

            middleware = falcon.CacheMiddleware(
                ttl=60,  # 1 minute
            )
            app = falcon.App(middleware=[middleware])

        With a custom store (e.g., Redis)::

            redis_store = MyRedisCacheStore(redis_client)
            app = falcon.App(middleware=[
                falcon.CacheMiddleware(store=redis_store)
            ])

    Note:
        The cache key is based on the request path and query string.
        Different query parameters will result in different cache entries.

    Note:
        For ASGI applications, this middleware uses async storage operations
        via ``get_async``, ``set_async``, etc. methods on the store.

    Note:
        Cache invalidation is performed based on the request path only
        (without query string). This means a POST to ``/items`` will
        invalidate cache entries for ``/items``, ``/items?page=1``, etc.
    """

    def __init__(
        self,
        store: CacheStore | None = None,
        ttl: int = DEFAULT_CACHE_TTL,
        methods: Iterable[str] = ('GET',),
        invalidation_methods: Iterable[str] = ('POST', 'PUT', 'PATCH', 'DELETE'),
    ) -> None:
        self._store = store if store is not None else InMemoryCacheStore(ttl=ttl)
        self._ttl = ttl
        self._methods = frozenset(m.upper() for m in methods)
        self._invalidation_methods = frozenset(m.upper() for m in invalidation_methods)

    def _normalize_query_string(self, query_string: str) -> str:
        """Normalize a query string by sorting parameters alphabetically.

        This ensures that query strings with the same parameters but in different
        order produce the same cache key. For example, ``foo=1&bar=2`` and
        ``bar=2&foo=1`` will both normalize to ``bar=2&foo=1``.

        Args:
            query_string: The raw query string to normalize.

        Returns:
            The normalized query string with parameters sorted alphabetically.
        """
        if not query_string:
            return ''
        # Parse query string into list of (key, value) pairs
        params = urllib.parse.parse_qsl(query_string, keep_blank_values=True)
        # Sort by key, then by value for consistent ordering
        params.sort()
        # Re-encode to query string
        return urllib.parse.urlencode(params)

    def _create_cache_key(self, req: Request | AsgiRequest) -> str:
        """Create a unique cache key from request attributes.

        The cache key combines the request path and normalized query string to
        ensure uniqueness across different endpoints and query parameters.
        Query parameters are sorted alphabetically so that requests with the
        same parameters in different order will use the same cache entry.
        """
        query_string = req.query_string
        if query_string:
            normalized_query = self._normalize_query_string(query_string)
            if normalized_query:
                return f'{req.path}?{normalized_query}'
        return req.path

    def _create_invalidation_prefix(self, req: Request | AsgiRequest) -> str:
        """Create the prefix to use for cache invalidation.

        This returns just the path so that all cache entries for that path
        (regardless of query string) are invalidated.
        """
        return req.path

    def _restore_response(
        self, resp: Response | AsgiResponse, cached: CachedResponse
    ) -> None:
        """Restore a cached response to the response object."""
        resp.status = cached.status
        for name, value in cached.headers.items():
            resp.set_header(name, value)
        if cached.media is not None:
            resp.media = cached.media
        elif cached.data is not None:
            resp.data = cached.data
        elif cached.text is not None:
            resp.text = cached.text

    def _normalize_status(self, status: str | int) -> str:
        """Normalize status to string format.

        Args:
            status: The status code or string.

        Returns:
            A status string like '200 OK' or '201 Created'.
        """
        if isinstance(status, int):
            return code_to_http_status(status)
        return str(status)

    def _capture_response(self, resp: Response | AsgiResponse) -> CachedResponse:
        """Capture the current response state for caching.

        Note:
            This method uses the public ``media``, ``data``, and ``text``
            properties of the Response object to capture the response state.
        """
        return CachedResponse(
            status=self._normalize_status(resp.status),
            headers=resp.headers,
            media=resp.media,
            data=resp.data,
            text=resp.text,
            timestamp=time.time(),
        )

    def process_request(self, req: Request, resp: Response) -> None:
        """Process the request before routing (WSGI).

        For cacheable methods (GET by default), checks if a cached response
        exists and returns it if found.

        For invalidation methods (POST, PUT, PATCH, DELETE by default),
        invalidates any cached entries for the request path.
        """
        # Handle cache invalidation for modifying requests
        if req.method in self._invalidation_methods:
            prefix = self._create_invalidation_prefix(req)
            self._store.invalidate_prefix(prefix)
            return

        # Only cache configured methods
        if req.method not in self._methods:
            return

        cache_key = self._create_cache_key(req)
        cached = self._store.get(cache_key)

        if cached is not None:
            self._restore_response(resp, cached)
            resp.complete = True
            return

        # Mark that we need to cache this response
        req.context._cache_key = cache_key

    def process_response(
        self, req: Request, resp: Response, resource: object, req_succeeded: bool
    ) -> None:
        """Process the response after routing (WSGI).

        If the request was cacheable and succeeded, cache the response.
        """
        cache_key = getattr(req.context, '_cache_key', None)
        if cache_key is None:
            return

        # Only cache successful responses
        if not req_succeeded:
            return

        cached_response = self._capture_response(resp)
        self._store.set(cache_key, cached_response, self._ttl)

    async def process_request_async(
        self, req: AsgiRequest, resp: AsgiResponse
    ) -> None:
        """Process the request before routing (ASGI).

        For cacheable methods (GET by default), checks if a cached response
        exists and returns it if found.

        For invalidation methods (POST, PUT, PATCH, DELETE by default),
        invalidates any cached entries for the request path.
        """
        # Handle cache invalidation for modifying requests
        if req.method in self._invalidation_methods:
            prefix = self._create_invalidation_prefix(req)
            await self._store.invalidate_prefix_async(prefix)
            return

        # Only cache configured methods
        if req.method not in self._methods:
            return

        cache_key = self._create_cache_key(req)
        cached = await self._store.get_async(cache_key)

        if cached is not None:
            self._restore_response(resp, cached)
            resp.complete = True
            return

        # Mark that we need to cache this response
        req.context._cache_key = cache_key

    async def process_response_async(
        self,
        req: AsgiRequest,
        resp: AsgiResponse,
        resource: object,
        req_succeeded: bool,
    ) -> None:
        """Process the response after routing (ASGI).

        If the request was cacheable and succeeded, cache the response.
        """
        cache_key = getattr(req.context, '_cache_key', None)
        if cache_key is None:
            return

        # Only cache successful responses
        if not req_succeeded:
            return

        cached_response = self._capture_response(resp)
        await self._store.set_async(cache_key, cached_response, self._ttl)


class TimeoutMiddleware:
    """Request Processing Timeout Middleware.

    This middleware enforces a timeout on request processing. When a request
    takes longer than the configured timeout, a 503 Service Unavailable
    response is returned via :class:`~falcon.HTTPRequestProcessingTimeout`.

    Note:
        **WSGI vs ASGI behavior**:

        - **WSGI (synchronous)**: Timeout detection is "best effort" only.
          The middleware checks elapsed time at middleware boundaries but cannot
          actually interrupt running Python code. This serves as a safety net
          and provides timeout reporting after the fact. For true timeout
          enforcement in WSGI apps, use server-level timeouts (e.g., Gunicorn's
          ``timeout`` setting).

        - **ASGI (asynchronous)**: Timeout detection at middleware boundaries,
          similar to WSGI. The elapsed time is checked before resource execution
          and after response completion.

    Keyword Arguments:
        default_timeout (float): Default timeout limit in seconds for all requests.
            Set to ``None`` to disable timeout by default. Defaults to 30.0.
        timeout_resolver (callable): Optional function that returns a custom
            timeout limit for each request. Signature: ``func(req, resp, resource)``
            returning a float (timeout limit in seconds) or ``None`` (no timeout).
            This is called after routing but before resource execution.
            When provided, this overrides ``default_timeout`` for requests
            where it returns a non-None value.

            Warning:
                The ``resource`` argument may be ``None`` in cases where routing
                fails or no resource is matched (e.g., 404 responses). Your
                resolver function should handle this case gracefully.

        include_response_time (bool): If ``True``, add an ``X-Response-Time``
            header showing actual processing time in milliseconds.
            Defaults to ``False``.

    Example:
        Basic usage with default timeout::

            import falcon

            app = falcon.App(middleware=[
                falcon.TimeoutMiddleware(default_timeout=30.0)
            ])

        With per-route timeouts::

            def get_timeout(req, resp, resource):
                # Longer timeout for specific paths
                if req.path.startswith('/slow'):
                    return 120.0  # 2 minutes
                if req.path.startswith('/fast'):
                    return 5.0
                return None  # Use default

            middleware = falcon.TimeoutMiddleware(
                default_timeout=30.0,
                timeout_resolver=get_timeout,
            )
            app = falcon.App(middleware=[middleware])

        Resource-based timeouts::

            def get_timeout(req, resp, resource):
                return getattr(resource, 'timeout', None)

            class SlowResource:
                timeout = 60.0

                def on_get(self, req, resp):
                    # This route has a 60s timeout
                    pass

            middleware = falcon.TimeoutMiddleware(
                default_timeout=30.0,
                timeout_resolver=get_timeout,
            )

    .. versionadded:: 4.1
    """

    def __init__(
        self,
        default_timeout: float | None = DEFAULT_TIMEOUT,
        timeout_resolver: Callable[
            [Request | AsgiRequest, Response | AsgiResponse, object | None],
            float | None,
        ]
        | None = None,
        include_response_time: bool = False,
    ) -> None:
        self._default_timeout = default_timeout
        self._timeout_resolver = timeout_resolver
        self._include_response_time = include_response_time

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _get_timeout(
        self,
        req: Request | AsgiRequest,
        resp: Response | AsgiResponse,
        resource: object | None,
    ) -> float | None:
        """Determine the timeout limit for this request.

        Args:
            req: The request object.
            resp: The response object.
            resource: The matched resource object, or ``None`` if no resource
                was matched (e.g., 404 responses).

        Returns:
            The timeout limit in seconds, or ``None`` if timeout is disabled.
        """
        if self._timeout_resolver is not None:
            custom_timeout = self._timeout_resolver(req, resp, resource)
            if custom_timeout is not None:
                return custom_timeout
        return self._default_timeout

    def _record_start_time(self, req: Request | AsgiRequest) -> None:
        """Record the request start time in the request context."""
        req.context._timeout_start = time.monotonic()

    def _check_and_store_timeout(
        self,
        req: Request | AsgiRequest,
        resp: Response | AsgiResponse,
        resource: object,
    ) -> None:
        """Determine timeout limit, store it, and check if already exceeded.

        This method is called during the resource phase (after routing but
        before resource execution). It determines the appropriate timeout limit,
        stores it in the request context, and raises an error if the timeout
        has already been exceeded.

        Raises:
            HTTPRequestProcessingTimeout: If the elapsed time already exceeds
                the configured timeout limit.
        """
        timeout_limit = self._get_timeout(req, resp, resource)
        req.context._timeout_limit = timeout_limit

        if timeout_limit is not None:
            start_time = getattr(req.context, '_timeout_start', None)
            if start_time is not None:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout_limit:
                    self._raise_timeout_error(timeout_limit, elapsed)

    def _check_response_timeout(
        self,
        req: Request | AsgiRequest,
        resp: Response | AsgiResponse,
        req_succeeded: bool,
    ) -> None:
        """Check elapsed time and optionally add response time header.

        This method is called during the response phase. It calculates the
        total elapsed time, optionally adds a response time header, and
        raises a timeout error if the request exceeded the configured timeout limit.

        Args:
            req: The request object.
            resp: The response object.
            req_succeeded: Whether the request succeeded from the middleware
                perspective. Timeout errors are only raised for successful
                requests to avoid overriding existing error responses.

        Raises:
            HTTPRequestProcessingTimeout: If the request succeeded but exceeded
                the configured timeout limit.
        """
        start_time = getattr(req.context, '_timeout_start', None)
        if start_time is None:
            return

        elapsed = time.monotonic() - start_time

        if self._include_response_time:
            resp.set_header('X-Response-Time', f'{elapsed * 1000:.2f}ms')

        timeout_limit = getattr(req.context, '_timeout_limit', None)
        if timeout_limit is not None and elapsed > timeout_limit and req_succeeded:
            self._raise_timeout_error(timeout_limit, elapsed)

    def _raise_timeout_error(self, timeout_limit: float, elapsed: float) -> None:
        """Raise a timeout error with a consistent, detailed message.

        Args:
            timeout_limit: The configured timeout limit in seconds.
            elapsed: The actual elapsed time in seconds.

        Raises:
            HTTPRequestProcessingTimeout: Always raised with detailed message.
        """
        raise HTTPRequestProcessingTimeout(
            timeout=timeout_limit,
            description=(
                f'Request processing exceeded the {timeout_limit:.1f}s '
                f'timeout limit (actual: {elapsed:.1f}s).'
            ),
        )

    # -------------------------------------------------------------------------
    # WSGI Methods
    # -------------------------------------------------------------------------

    def process_request(self, req: Request, resp: Response) -> None:
        """Record the request start time (WSGI)."""
        self._record_start_time(req)

    def process_resource(
        self,
        req: Request,
        resp: Response,
        resource: object,
        params: dict[str, Any],
    ) -> None:
        """Determine and store timeout for this request (WSGI).

        Also checks if the timeout has already been exceeded before
        resource execution.
        """
        self._check_and_store_timeout(req, resp, resource)

    def process_response(
        self,
        req: Request,
        resp: Response,
        resource: object,
        req_succeeded: bool,
    ) -> None:
        """Check elapsed time and optionally add response time header (WSGI).

        If the request exceeded the configured timeout and completed
        successfully, raises :class:`~falcon.HTTPRequestProcessingTimeout`.
        """
        self._check_response_timeout(req, resp, req_succeeded)

    # -------------------------------------------------------------------------
    # ASGI Methods
    # -------------------------------------------------------------------------

    async def process_request_async(
        self, req: AsgiRequest, resp: AsgiResponse
    ) -> None:
        """Record the request start time (ASGI)."""
        self._record_start_time(req)

    async def process_resource_async(
        self,
        req: AsgiRequest,
        resp: AsgiResponse,
        resource: object,
        params: dict[str, Any],
    ) -> None:
        """Determine and store timeout for this request (ASGI).

        Also checks if the timeout has already been exceeded before
        resource execution.
        """
        self._check_and_store_timeout(req, resp, resource)

    async def process_response_async(
        self,
        req: AsgiRequest,
        resp: AsgiResponse,
        resource: object,
        req_succeeded: bool,
    ) -> None:
        """Check elapsed time and optionally add response time header (ASGI).

        If the request exceeded the configured timeout and completed
        successfully, raises :class:`~falcon.HTTPRequestProcessingTimeout`.
        """
        self._check_response_timeout(req, resp, req_succeeded)
