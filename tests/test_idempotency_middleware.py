import time

import pytest

import falcon
from falcon import testing
from falcon.middleware import (
    CachedResponse,
    IdempotencyMiddleware,
    InMemoryIdempotencyStore,
)


class CreateResource:
    """Resource that simulates creating an item."""

    call_count = 0

    def on_post(self, req, resp):
        CreateResource.call_count += 1
        resp.media = {'id': CreateResource.call_count, 'created': True}
        resp.status = falcon.HTTP_201

    def on_put(self, req, resp):
        CreateResource.call_count += 1
        resp.media = {'id': CreateResource.call_count, 'updated': True}

    def on_patch(self, req, resp):
        CreateResource.call_count += 1
        resp.media = {'id': CreateResource.call_count, 'patched': True}

    def on_get(self, req, resp):
        CreateResource.call_count += 1
        resp.media = {'id': CreateResource.call_count, 'fetched': True}


class CreateResourceAsync:
    """Async resource that simulates creating an item."""

    call_count = 0

    async def on_post(self, req, resp):
        CreateResourceAsync.call_count += 1
        resp.media = {'id': CreateResourceAsync.call_count, 'created': True}
        resp.status = falcon.HTTP_201

    async def on_put(self, req, resp):
        CreateResourceAsync.call_count += 1
        resp.media = {'id': CreateResourceAsync.call_count, 'updated': True}

    async def on_patch(self, req, resp):
        CreateResourceAsync.call_count += 1
        resp.media = {'id': CreateResourceAsync.call_count, 'patched': True}

    async def on_get(self, req, resp):
        CreateResourceAsync.call_count += 1
        resp.media = {'id': CreateResourceAsync.call_count, 'fetched': True}


class ErrorResource:
    """Resource that raises an error."""

    def on_post(self, req, resp):
        raise falcon.HTTPBadRequest(description='Invalid request')


class ErrorResourceAsync:
    """Async resource that raises an error."""

    async def on_post(self, req, resp):
        raise falcon.HTTPBadRequest(description='Invalid request')


class TextResource:
    """Resource that returns text."""

    def on_post(self, req, resp):
        resp.text = 'Hello, World!'
        resp.content_type = 'text/plain'


class DataResource:
    """Resource that returns raw bytes."""

    def on_post(self, req, resp):
        resp.data = b'binary data'
        resp.content_type = 'application/octet-stream'


class HeaderResource:
    """Resource that sets custom headers."""

    def on_post(self, req, resp):
        resp.set_header('X-Custom-Header', 'custom-value')
        resp.set_header('X-Request-Id', '12345')
        resp.media = {'success': True}


@pytest.fixture
def make_client(asgi, util):
    """Factory fixture to create test clients with idempotency middleware."""

    def _make_client(middleware=None, **middleware_kwargs):
        if middleware is None:
            middleware = IdempotencyMiddleware(**middleware_kwargs)
        app = util.create_app(asgi, middleware=[middleware])
        return testing.TestClient(app)

    return _make_client


@pytest.fixture(autouse=True)
def reset_counters():
    """Reset resource call counters before each test."""
    CreateResource.call_count = 0
    CreateResourceAsync.call_count = 0
    yield


class TestIdempotencyMiddleware:
    """Tests for IdempotencyMiddleware."""

    def test_no_idempotency_key_processes_normally(self, make_client, asgi):
        client = make_client()
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        # First request
        result1 = client.simulate_post('/items')
        assert result1.status == falcon.HTTP_201
        assert result1.json['id'] == 1

        # Second request without key - should process again
        result2 = client.simulate_post('/items')
        assert result2.status == falcon.HTTP_201
        assert result2.json['id'] == 2

    def test_idempotency_key_returns_cached_response(self, make_client, asgi):
        client = make_client()
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        idempotency_key = 'unique-key-123'

        # First request with idempotency key
        result1 = client.simulate_post(
            '/items', headers={'Idempotency-Key': idempotency_key}
        )
        assert result1.status == falcon.HTTP_201
        assert result1.json['id'] == 1

        # Second request with same key - should return cached response
        result2 = client.simulate_post(
            '/items', headers={'Idempotency-Key': idempotency_key}
        )
        assert result2.status == falcon.HTTP_201
        assert result2.json['id'] == 1  # Same ID as first request

        # Verify the resource was only called once
        count = CreateResourceAsync.call_count if asgi else CreateResource.call_count
        assert count == 1

    def test_different_keys_process_separately(self, make_client, asgi):
        client = make_client()
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        # First request with key A
        result1 = client.simulate_post('/items', headers={'Idempotency-Key': 'key-A'})
        assert result1.json['id'] == 1

        # Second request with key B - should process as new request
        result2 = client.simulate_post('/items', headers={'Idempotency-Key': 'key-B'})
        assert result2.json['id'] == 2

        # Verify both requests were processed
        count = CreateResourceAsync.call_count if asgi else CreateResource.call_count
        assert count == 2

    def test_same_key_different_paths_process_separately(self, make_client, asgi):
        client = make_client()
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)
        client.app.add_route('/other', resource)

        key = 'same-key'

        # Request to /items
        result1 = client.simulate_post('/items', headers={'Idempotency-Key': key})
        assert result1.json['id'] == 1

        # Request to /other with same key - should process as new (different path)
        result2 = client.simulate_post('/other', headers={'Idempotency-Key': key})
        assert result2.json['id'] == 2

    def test_same_key_different_methods_process_separately(self, make_client, asgi):
        client = make_client()
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        key = 'same-key'

        # POST request
        result1 = client.simulate_post('/items', headers={'Idempotency-Key': key})
        assert result1.json['created'] is True

        # PUT request with same key - should process as new (different method)
        result2 = client.simulate_put('/items', headers={'Idempotency-Key': key})
        assert result2.json['updated'] is True

    def test_get_requests_not_cached_by_default(self, make_client, asgi):
        client = make_client()
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        key = 'get-key'

        # Two GET requests with same key
        result1 = client.simulate_get('/items', headers={'Idempotency-Key': key})
        result2 = client.simulate_get('/items', headers={'Idempotency-Key': key})

        # Both should be processed (GET not cached by default)
        assert result1.json['id'] == 1
        assert result2.json['id'] == 2

    def test_custom_methods_configuration(self, make_client, asgi):
        # Include GET in idempotent methods
        client = make_client(methods=['GET', 'POST'])
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        key = 'get-key'

        # Two GET requests with same key
        result1 = client.simulate_get('/items', headers={'Idempotency-Key': key})
        result2 = client.simulate_get('/items', headers={'Idempotency-Key': key})

        # Second should return cached response
        assert result1.json['id'] == 1
        assert result2.json['id'] == 1

    def test_custom_header_name(self, make_client, asgi):
        client = make_client(header_name='X-Request-Id')
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        # Request with custom header
        result1 = client.simulate_post('/items', headers={'X-Request-Id': 'req-123'})
        result2 = client.simulate_post('/items', headers={'X-Request-Id': 'req-123'})

        assert result1.json['id'] == 1
        assert result2.json['id'] == 1

    def test_failed_request_not_cached(self, make_client, asgi):
        client = make_client()
        resource = ErrorResourceAsync() if asgi else ErrorResource()
        client.app.add_route('/items', resource)

        key = 'error-key'

        # First request fails
        result1 = client.simulate_post('/items', headers={'Idempotency-Key': key})
        assert result1.status == falcon.HTTP_400

        # Replace with working resource
        working_resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', working_resource)

        # Second request should process (error was not cached)
        result2 = client.simulate_post('/items', headers={'Idempotency-Key': key})
        assert result2.status == falcon.HTTP_201

    def test_text_response_cached(self, make_client, asgi):
        client = make_client()
        client.app.add_route('/text', TextResource())

        key = 'text-key'

        result1 = client.simulate_post('/text', headers={'Idempotency-Key': key})
        result2 = client.simulate_post('/text', headers={'Idempotency-Key': key})

        assert result1.text == 'Hello, World!'
        assert result2.text == 'Hello, World!'

    def test_data_response_cached(self, make_client, asgi):
        client = make_client()
        client.app.add_route('/data', DataResource())

        key = 'data-key'

        result1 = client.simulate_post('/data', headers={'Idempotency-Key': key})
        result2 = client.simulate_post('/data', headers={'Idempotency-Key': key})

        assert result1.content == b'binary data'
        assert result2.content == b'binary data'

    def test_headers_cached(self, make_client, asgi):
        client = make_client()
        client.app.add_route('/headers', HeaderResource())

        key = 'header-key'

        result1 = client.simulate_post('/headers', headers={'Idempotency-Key': key})
        result2 = client.simulate_post('/headers', headers={'Idempotency-Key': key})

        assert result1.headers.get('X-Custom-Header') == 'custom-value'
        assert result2.headers.get('X-Custom-Header') == 'custom-value'
        assert result1.headers.get('X-Request-Id') == '12345'
        assert result2.headers.get('X-Request-Id') == '12345'


class TestInMemoryIdempotencyStore:
    """Tests for InMemoryIdempotencyStore."""

    def test_get_nonexistent_key(self):
        store = InMemoryIdempotencyStore()
        assert store.get('nonexistent') is None

    def test_set_and_get(self):
        store = InMemoryIdempotencyStore()
        cached = CachedResponse(
            status='200 OK',
            headers={'content-type': 'application/json'},
            media={'test': 'data'},
            data=None,
            text=None,
            timestamp=time.time(),
        )

        store.set('test-key', cached)
        result = store.get('test-key')

        assert result is not None
        assert result.status == '200 OK'
        assert result.media == {'test': 'data'}

    def test_expired_entry_returns_none(self):
        store = InMemoryIdempotencyStore(ttl=1)  # 1 second TTL
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media=None,
            data=None,
            text=None,
            timestamp=time.time() - 2,  # 2 seconds ago
        )

        store.set('expired-key', cached)
        assert store.get('expired-key') is None

    def test_cleanup_on_threshold(self):
        store = InMemoryIdempotencyStore(ttl=1)

        # Add old entries
        old_timestamp = time.time() - 10
        for i in range(1005):
            cached = CachedResponse(
                status='200 OK',
                headers={},
                media=None,
                data=None,
                text=None,
                timestamp=old_timestamp,
            )
            store._cache[f'old-key-{i}'] = cached

        # Add a new entry which should trigger cleanup
        new_cached = CachedResponse(
            status='200 OK',
            headers={},
            media=None,
            data=None,
            text=None,
            timestamp=time.time(),
        )
        store.set('new-key', new_cached)

        # Old entries should be cleaned up
        assert len(store._cache) < 1005

    @pytest.mark.asyncio
    async def test_async_get_and_set(self):
        store = InMemoryIdempotencyStore()
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media={'async': 'test'},
            data=None,
            text=None,
            timestamp=time.time(),
        )

        await store.set_async('async-key', cached)
        result = await store.get_async('async-key')

        assert result is not None
        assert result.media == {'async': 'test'}


class TestIdempotencyMiddlewareIntegration:
    """Integration tests for IdempotencyMiddleware with custom stores."""

    def test_custom_store(self, asgi, util):
        """Test using a custom store implementation."""

        class CustomStore:
            def __init__(self):
                self._data = {}

            def get(self, key):
                return self._data.get(key)

            def set(self, key, response, ttl=None):
                self._data[key] = response

            async def get_async(self, key):
                return self.get(key)

            async def set_async(self, key, response, ttl=None):
                self.set(key, response, ttl)

        custom_store = CustomStore()
        middleware = IdempotencyMiddleware(store=custom_store)
        app = util.create_app(asgi, middleware=[middleware])
        client = testing.TestClient(app)

        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        key = 'custom-store-key'
        result1 = client.simulate_post('/items', headers={'Idempotency-Key': key})
        result2 = client.simulate_post('/items', headers={'Idempotency-Key': key})

        assert result1.json['id'] == 1
        assert result2.json['id'] == 1

    def test_middleware_with_cors(self, asgi, util):
        """Test that IdempotencyMiddleware works alongside CORSMiddleware."""
        idempotency = IdempotencyMiddleware()
        cors = falcon.CORSMiddleware()
        app = util.create_app(asgi, middleware=[idempotency, cors])
        client = testing.TestClient(app)

        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        key = 'cors-test-key'
        result1 = client.simulate_post(
            '/items',
            headers={'Idempotency-Key': key, 'Origin': 'http://example.com'},
        )
        result2 = client.simulate_post(
            '/items',
            headers={'Idempotency-Key': key, 'Origin': 'http://example.com'},
        )

        assert result1.json['id'] == 1
        assert result2.json['id'] == 1
        assert result1.headers.get('Access-Control-Allow-Origin') == '*'


class TestCachedResponse:
    """Tests for CachedResponse dataclass."""

    def test_creation(self):
        cached = CachedResponse(
            status='201 Created',
            headers={'content-type': 'application/json'},
            media={'key': 'value'},
            data=None,
            text=None,
            timestamp=1234567890.0,
        )

        assert cached.status == '201 Created'
        assert cached.headers == {'content-type': 'application/json'}
        assert cached.media == {'key': 'value'}
        assert cached.data is None
        assert cached.text is None
        assert cached.timestamp == 1234567890.0

    def test_with_text(self):
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media=None,
            data=None,
            text='Hello',
            timestamp=time.time(),
        )

        assert cached.text == 'Hello'

    def test_with_data(self):
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media=None,
            data=b'binary',
            text=None,
            timestamp=time.time(),
        )

        assert cached.data == b'binary'


class TestQueryStringDifferentiation:
    """Tests for query string handling in cache keys."""

    def test_different_query_strings_cached_separately(self, make_client, asgi):
        """Different query strings should be treated as different requests."""
        client = make_client()
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        key = 'same-key'

        # Request with query foo=1
        result1 = client.simulate_post(
            '/items', params={'foo': '1'}, headers={'Idempotency-Key': key}
        )
        # Request with query foo=2
        result2 = client.simulate_post(
            '/items', params={'foo': '2'}, headers={'Idempotency-Key': key}
        )

        # Different query strings should produce different responses
        assert result1.json['id'] == 1
        assert result2.json['id'] == 2

    def test_same_query_string_returns_cached(self, make_client, asgi):
        """Same query string should return cached response."""
        client = make_client()
        resource = CreateResourceAsync() if asgi else CreateResource()
        client.app.add_route('/items', resource)

        key = 'same-key'

        # Two requests with same query
        result1 = client.simulate_post(
            '/items', params={'foo': '1'}, headers={'Idempotency-Key': key}
        )
        result2 = client.simulate_post(
            '/items', params={'foo': '1'}, headers={'Idempotency-Key': key}
        )

        # Same query string should return cached response
        assert result1.json['id'] == 1
        assert result2.json['id'] == 1


class TestStatusNormalization:
    """Tests for status normalization."""

    def test_integer_status_normalized_to_string(self, make_client, asgi):
        """Integer status should be normalized to string format."""

        class IntStatusResource:
            def on_post(self, req, resp):
                resp.status = 201  # Integer status
                resp.media = {'created': True}

        class IntStatusResourceAsync:
            async def on_post(self, req, resp):
                resp.status = 201  # Integer status
                resp.media = {'created': True}

        client = make_client()
        resource = IntStatusResourceAsync() if asgi else IntStatusResource()
        client.app.add_route('/items', resource)

        key = 'status-key'

        result1 = client.simulate_post('/items', headers={'Idempotency-Key': key})
        result2 = client.simulate_post('/items', headers={'Idempotency-Key': key})

        # Both should have string status
        assert '201' in result1.status
        assert '201' in result2.status
        assert 'Created' in result1.status
        assert 'Created' in result2.status


class TestLockingBehavior:
    """Tests for race condition protection."""

    def test_acquire_or_wait_returns_cached_when_available(self):
        """acquire_or_wait should return cached response if available."""
        store = InMemoryIdempotencyStore()
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media={'cached': True},
            data=None,
            text=None,
            timestamp=time.time(),
        )

        # Set a cached response
        store.set('test-key', cached)

        # acquire_or_wait should return the cached response
        result = store.acquire_or_wait('test-key')
        assert result is not None
        assert result.media == {'cached': True}

    def test_acquire_or_wait_returns_none_for_new_key(self):
        """acquire_or_wait should return None and acquire lock for new key."""
        store = InMemoryIdempotencyStore()

        # First call should return None (acquired lock)
        result = store.acquire_or_wait('new-key')
        assert result is None

        # Key should be marked as processing
        from falcon.middleware import _PROCESSING

        assert store._cache.get('new-key') is _PROCESSING

    def test_release_removes_processing_marker(self):
        """release should remove the processing marker."""
        store = InMemoryIdempotencyStore()

        # Acquire lock
        store.acquire_or_wait('release-key')

        # Release it
        store.release('release-key')

        # Key should be removed
        assert 'release-key' not in store._cache

    @pytest.mark.asyncio
    async def test_async_acquire_or_wait(self):
        """Async acquire_or_wait should work correctly."""
        store = InMemoryIdempotencyStore()

        # Acquire lock
        result = await store.acquire_or_wait_async('async-key')
        assert result is None

        # Set a response
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media={'async': True},
            data=None,
            text=None,
            timestamp=time.time(),
        )
        await store.set_async('async-key', cached)

        # Now acquire_or_wait should return the cached response
        result2 = await store.acquire_or_wait_async('async-key')
        assert result2 is not None
        assert result2.media == {'async': True}


class TestCacheKeyFormat:
    """Tests for cache key format."""

    def test_cache_key_without_query_string(self):
        """Cache key should not include '?' when there's no query string."""
        middleware = IdempotencyMiddleware()

        # Create a mock request with no query string
        import falcon.testing

        req = falcon.testing.create_req(method='POST', path='/items')
        cache_key = middleware._create_cache_key(req, 'test-key')

        assert cache_key == 'POST:/items:test-key'
        assert '?' not in cache_key

    def test_cache_key_with_query_string(self):
        """Cache key should include '?' only when there's a query string."""
        middleware = IdempotencyMiddleware()

        import falcon.testing

        req = falcon.testing.create_req(
            method='POST', path='/items', query_string='foo=bar'
        )
        cache_key = middleware._create_cache_key(req, 'test-key')

        assert cache_key == 'POST:/items?foo=bar:test-key'
        assert '?' in cache_key


class TestConcurrentRequests:
    """Tests for concurrent request handling with race condition protection."""

    def test_concurrent_requests_wsgi_only_one_handler_executes(self):
        """Multiple concurrent requests with same key should only execute handler once.

        This test verifies:
        1. Only one handler execution happens for concurrent requests with same key
        2. All requests receive the same response
        3. No requests get stuck waiting indefinitely
        """
        import threading

        class SlowResource:
            call_count = 0
            call_lock = threading.Lock()

            def on_post(self, req, resp):
                with SlowResource.call_lock:
                    SlowResource.call_count += 1
                    current_count = SlowResource.call_count
                # Simulate slow processing
                time.sleep(0.2)
                resp.media = {'id': current_count, 'message': 'created'}
                resp.status = falcon.HTTP_201

        # Reset counter
        SlowResource.call_count = 0

        app = falcon.App(middleware=[IdempotencyMiddleware()])
        app.add_route('/items', SlowResource())

        results = []
        errors = []
        completion_times = []

        def make_request():
            try:
                start_time = time.time()
                client = testing.TestClient(app)
                result = client.simulate_post(
                    '/items', headers={'Idempotency-Key': 'concurrent-key'}
                )
                elapsed = time.time() - start_time
                results.append((result.status, result.json))
                completion_times.append(elapsed)
            except Exception as e:
                errors.append(str(e))

        # Start 10 concurrent requests with the same key
        threads = []
        for _ in range(10):
            t = threading.Thread(target=make_request)
            threads.append(t)

        start = time.time()
        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=10.0)  # 10 second timeout per thread

        total_time = time.time() - start

        # Verify no errors
        assert len(errors) == 0, f'Errors occurred: {errors}'

        # Verify all requests completed
        assert len(results) == 10, f'Expected 10 results, got {len(results)}'

        # Verify handler was called only once
        assert SlowResource.call_count == 1, (
            f'Handler should be called once, but was called {SlowResource.call_count} times'
        )

        # Verify all requests got the same response
        statuses = [r[0] for r in results]
        responses = [r[1] for r in results]
        assert all(s == falcon.HTTP_201 for s in statuses), f'Not all statuses are 201: {statuses}'
        assert all(r == {'id': 1, 'message': 'created'} for r in responses), (
            f'Not all responses match: {responses}'
        )

        # Verify requests didn't get stuck (total time should be reasonable)
        # First request takes ~0.2s, others should return quickly after that
        assert total_time < 5.0, f'Requests took too long: {total_time}s'

    def test_concurrent_requests_different_keys_process_independently(self):
        """Concurrent requests with different keys should process independently."""
        import threading

        class CountingResource:
            call_count = 0
            call_lock = threading.Lock()

            def on_post(self, req, resp):
                with CountingResource.call_lock:
                    CountingResource.call_count += 1
                    current_count = CountingResource.call_count
                time.sleep(0.05)  # Small delay
                resp.media = {'id': current_count}

        CountingResource.call_count = 0

        app = falcon.App(middleware=[IdempotencyMiddleware()])
        app.add_route('/items', CountingResource())

        results = {}
        errors = []

        def make_request(key):
            try:
                client = testing.TestClient(app)
                result = client.simulate_post(
                    '/items', headers={'Idempotency-Key': key}
                )
                results[key] = result.json
            except Exception as e:
                errors.append(str(e))

        # Start requests with different keys
        threads = []
        keys = [f'key-{i}' for i in range(5)]
        for key in keys:
            t = threading.Thread(target=make_request, args=(key,))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f'Errors occurred: {errors}'
        assert CountingResource.call_count == 5, (
            f'Expected 5 handler calls, got {CountingResource.call_count}'
        )
        # Each key should have a unique id
        ids = [results[k]['id'] for k in keys]
        assert len(set(ids)) == 5, f'Expected 5 unique ids, got {ids}'

    def test_subsequent_requests_after_concurrent_batch(self):
        """Requests after initial concurrent batch should also get cached response."""
        import threading

        class Resource:
            call_count = 0

            def on_post(self, req, resp):
                Resource.call_count += 1
                time.sleep(0.1)
                resp.media = {'id': Resource.call_count}

        Resource.call_count = 0

        app = falcon.App(middleware=[IdempotencyMiddleware()])
        app.add_route('/items', Resource())

        # First batch of concurrent requests
        results_batch1 = []

        def make_request(results_list):
            client = testing.TestClient(app)
            result = client.simulate_post(
                '/items', headers={'Idempotency-Key': 'batch-key'}
            )
            results_list.append(result.json)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=make_request, args=(results_batch1,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Verify first batch
        assert Resource.call_count == 1
        assert all(r == {'id': 1} for r in results_batch1)

        # Second batch of requests (should all get cached response immediately)
        results_batch2 = []
        threads2 = []
        for _ in range(5):
            t = threading.Thread(target=make_request, args=(results_batch2,))
            threads2.append(t)

        start = time.time()
        for t in threads2:
            t.start()
        for t in threads2:
            t.join(timeout=5.0)
        elapsed = time.time() - start

        # Verify second batch
        assert Resource.call_count == 1, 'Handler should not be called again'
        assert all(r == {'id': 1} for r in results_batch2)
        # Second batch should be fast (all from cache)
        assert elapsed < 0.5, f'Second batch took too long: {elapsed}s'

    def test_error_releases_lock_for_subsequent_requests(self):
        """If first request fails, lock should be released for retry."""
        import threading

        class FailOnceResource:
            call_count = 0
            call_lock = threading.Lock()

            def on_post(self, req, resp):
                with FailOnceResource.call_lock:
                    FailOnceResource.call_count += 1
                    count = FailOnceResource.call_count

                if count == 1:
                    raise falcon.HTTPInternalServerError(description='First call fails')

                resp.media = {'id': count, 'success': True}

        FailOnceResource.call_count = 0

        app = falcon.App(middleware=[IdempotencyMiddleware()])
        app.add_route('/items', FailOnceResource())

        # First request should fail
        client = testing.TestClient(app)
        result1 = client.simulate_post(
            '/items', headers={'Idempotency-Key': 'error-retry-key'}
        )
        assert result1.status == falcon.HTTP_500

        # Retry should succeed (lock was released)
        result2 = client.simulate_post(
            '/items', headers={'Idempotency-Key': 'error-retry-key'}
        )
        assert result2.status == falcon.HTTP_200
        assert result2.json['success'] is True

        # Third request should get cached response from second
        result3 = client.simulate_post(
            '/items', headers={'Idempotency-Key': 'error-retry-key'}
        )
        assert result3.status == falcon.HTTP_200
        assert result3.json == result2.json

        # Handler should be called exactly twice (first fails, second succeeds)
        assert FailOnceResource.call_count == 2
