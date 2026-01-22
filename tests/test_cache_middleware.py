import time

import pytest

import falcon
from falcon import testing
from falcon.middleware import (
    CachedResponse,
    CacheMiddleware,
    InMemoryCacheStore,
)


class GetResource:
    """Resource that simulates fetching data."""

    call_count = 0

    def on_get(self, req, resp):
        GetResource.call_count += 1
        resp.media = {'id': GetResource.call_count, 'fetched': True}
        resp.status = falcon.HTTP_200

    def on_post(self, req, resp):
        GetResource.call_count += 1
        resp.media = {'id': GetResource.call_count, 'created': True}
        resp.status = falcon.HTTP_201

    def on_put(self, req, resp):
        GetResource.call_count += 1
        resp.media = {'id': GetResource.call_count, 'updated': True}
        resp.status = falcon.HTTP_200

    def on_patch(self, req, resp):
        GetResource.call_count += 1
        resp.media = {'id': GetResource.call_count, 'patched': True}
        resp.status = falcon.HTTP_200

    def on_delete(self, req, resp):
        GetResource.call_count += 1
        resp.status = falcon.HTTP_204


class GetResourceAsync:
    """Async resource that simulates fetching data."""

    call_count = 0

    async def on_get(self, req, resp):
        GetResourceAsync.call_count += 1
        resp.media = {'id': GetResourceAsync.call_count, 'fetched': True}
        resp.status = falcon.HTTP_200

    async def on_post(self, req, resp):
        GetResourceAsync.call_count += 1
        resp.media = {'id': GetResourceAsync.call_count, 'created': True}
        resp.status = falcon.HTTP_201

    async def on_put(self, req, resp):
        GetResourceAsync.call_count += 1
        resp.media = {'id': GetResourceAsync.call_count, 'updated': True}
        resp.status = falcon.HTTP_200

    async def on_patch(self, req, resp):
        GetResourceAsync.call_count += 1
        resp.media = {'id': GetResourceAsync.call_count, 'patched': True}
        resp.status = falcon.HTTP_200

    async def on_delete(self, req, resp):
        GetResourceAsync.call_count += 1
        resp.status = falcon.HTTP_204


class ErrorResource:
    """Resource that raises an error."""

    def on_get(self, req, resp):
        raise falcon.HTTPBadRequest(description='Invalid request')


class ErrorResourceAsync:
    """Async resource that raises an error."""

    async def on_get(self, req, resp):
        raise falcon.HTTPBadRequest(description='Invalid request')


class TextResource:
    """Resource that returns text."""

    def on_get(self, req, resp):
        resp.text = 'Hello, World!'
        resp.content_type = 'text/plain'


class DataResource:
    """Resource that returns raw bytes."""

    def on_get(self, req, resp):
        resp.data = b'binary data'
        resp.content_type = 'application/octet-stream'


class HeaderResource:
    """Resource that sets custom headers."""

    def on_get(self, req, resp):
        resp.set_header('X-Custom-Header', 'custom-value')
        resp.set_header('X-Request-Id', '12345')
        resp.media = {'success': True}


@pytest.fixture
def make_client(asgi, util):
    """Factory fixture to create test clients with cache middleware."""

    def _make_client(middleware=None, **middleware_kwargs):
        if middleware is None:
            middleware = CacheMiddleware(**middleware_kwargs)
        app = util.create_app(asgi, middleware=[middleware])
        return testing.TestClient(app)

    return _make_client


@pytest.fixture(autouse=True)
def reset_counters():
    """Reset resource call counters before each test."""
    GetResource.call_count = 0
    GetResourceAsync.call_count = 0
    yield


class TestCacheMiddleware:
    """Tests for CacheMiddleware."""

    def test_get_request_cached(self, make_client, asgi):
        """GET requests should be cached."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # First request
        result1 = client.simulate_get('/items')
        assert result1.status == falcon.HTTP_200
        assert result1.json['id'] == 1

        # Second request should return cached response
        result2 = client.simulate_get('/items')
        assert result2.status == falcon.HTTP_200
        assert result2.json['id'] == 1  # Same ID as first request

        # Verify the resource was only called once
        count = GetResourceAsync.call_count if asgi else GetResource.call_count
        assert count == 1

    def test_post_request_not_cached(self, make_client, asgi):
        """POST requests should not be cached."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # Two POST requests should both execute
        result1 = client.simulate_post('/items')
        result2 = client.simulate_post('/items')

        assert result1.json['id'] == 1
        assert result2.json['id'] == 2

        count = GetResourceAsync.call_count if asgi else GetResource.call_count
        assert count == 2

    def test_post_invalidates_get_cache(self, make_client, asgi):
        """POST request should invalidate cached GET response for same path."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # First GET request - cached
        result1 = client.simulate_get('/items')
        assert result1.json['id'] == 1

        # POST request - invalidates cache
        client.simulate_post('/items')

        # Second GET request - should re-execute
        result2 = client.simulate_get('/items')
        assert result2.json['id'] == 3  # After POST (id=2), new GET is id=3

    def test_put_invalidates_cache(self, make_client, asgi):
        """PUT request should invalidate cached GET response."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # Cache a GET response
        result1 = client.simulate_get('/items')
        assert result1.json['id'] == 1

        # PUT invalidates cache
        client.simulate_put('/items')

        # GET should re-execute
        result2 = client.simulate_get('/items')
        assert result2.json['id'] == 3

    def test_patch_invalidates_cache(self, make_client, asgi):
        """PATCH request should invalidate cached GET response."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # Cache a GET response
        result1 = client.simulate_get('/items')
        assert result1.json['id'] == 1

        # PATCH invalidates cache
        client.simulate_patch('/items')

        # GET should re-execute
        result2 = client.simulate_get('/items')
        assert result2.json['id'] == 3

    def test_delete_invalidates_cache(self, make_client, asgi):
        """DELETE request should invalidate cached GET response."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # Cache a GET response
        result1 = client.simulate_get('/items')
        assert result1.json['id'] == 1

        # DELETE invalidates cache
        client.simulate_delete('/items')

        # GET should re-execute
        result2 = client.simulate_get('/items')
        assert result2.json['id'] == 3

    def test_different_paths_cached_separately(self, make_client, asgi):
        """Different paths should have separate cache entries."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)
        client.app.add_route('/other', resource)

        # GET /items
        result1 = client.simulate_get('/items')
        assert result1.json['id'] == 1

        # GET /other - should not use /items cache
        result2 = client.simulate_get('/other')
        assert result2.json['id'] == 2

        # GET /items again - should use cache
        result3 = client.simulate_get('/items')
        assert result3.json['id'] == 1

    def test_different_query_strings_cached_separately(self, make_client, asgi):
        """Different query strings should have separate cache entries."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # GET with query foo=1
        result1 = client.simulate_get('/items', params={'foo': '1'})
        assert result1.json['id'] == 1

        # GET with query foo=2 - different cache entry
        result2 = client.simulate_get('/items', params={'foo': '2'})
        assert result2.json['id'] == 2

        # GET with query foo=1 again - should use cache
        result3 = client.simulate_get('/items', params={'foo': '1'})
        assert result3.json['id'] == 1

    def test_post_invalidates_all_query_variants(self, make_client, asgi):
        """POST should invalidate all cached query string variants for a path."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # Cache multiple query variants
        client.simulate_get('/items')  # id=1
        client.simulate_get('/items', params={'page': '1'})  # id=2
        client.simulate_get('/items', params={'page': '2'})  # id=3

        # POST invalidates all
        client.simulate_post('/items')  # id=4

        # All GET variants should re-execute
        result1 = client.simulate_get('/items')
        assert result1.json['id'] == 5

        result2 = client.simulate_get('/items', params={'page': '1'})
        assert result2.json['id'] == 6

        result3 = client.simulate_get('/items', params={'page': '2'})
        assert result3.json['id'] == 7

    def test_failed_request_not_cached(self, make_client, asgi):
        """Failed GET requests should not be cached."""
        client = make_client()
        resource = ErrorResourceAsync() if asgi else ErrorResource()
        client.app.add_route('/items', resource)

        # First request fails
        result1 = client.simulate_get('/items')
        assert result1.status == falcon.HTTP_400

        # Replace with working resource
        working_resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', working_resource)

        # Second request should process (error was not cached)
        result2 = client.simulate_get('/items')
        assert result2.status == falcon.HTTP_200

    def test_text_response_cached(self, make_client, asgi):
        """Text responses should be cached correctly."""
        client = make_client()
        client.app.add_route('/text', TextResource())

        result1 = client.simulate_get('/text')
        result2 = client.simulate_get('/text')

        assert result1.text == 'Hello, World!'
        assert result2.text == 'Hello, World!'

    def test_data_response_cached(self, make_client, asgi):
        """Binary data responses should be cached correctly."""
        client = make_client()
        client.app.add_route('/data', DataResource())

        result1 = client.simulate_get('/data')
        result2 = client.simulate_get('/data')

        assert result1.content == b'binary data'
        assert result2.content == b'binary data'

    def test_headers_cached(self, make_client, asgi):
        """Response headers should be cached correctly."""
        client = make_client()
        client.app.add_route('/headers', HeaderResource())

        result1 = client.simulate_get('/headers')
        result2 = client.simulate_get('/headers')

        assert result1.headers.get('X-Custom-Header') == 'custom-value'
        assert result2.headers.get('X-Custom-Header') == 'custom-value'
        assert result1.headers.get('X-Request-Id') == '12345'
        assert result2.headers.get('X-Request-Id') == '12345'

    def test_custom_ttl(self, make_client, asgi):
        """Cache entries should expire after TTL."""
        store = InMemoryCacheStore(ttl=1)  # 1 second TTL
        client = make_client(store=store)
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # First request - cached
        result1 = client.simulate_get('/items')
        assert result1.json['id'] == 1

        # Wait for cache to expire
        time.sleep(1.1)

        # Second request - cache expired, should re-execute
        result2 = client.simulate_get('/items')
        assert result2.json['id'] == 2


class TestInMemoryCacheStore:
    """Tests for InMemoryCacheStore."""

    def test_get_nonexistent_key(self):
        """Getting a nonexistent key should return None."""
        store = InMemoryCacheStore()
        assert store.get('nonexistent') is None

    def test_set_and_get(self):
        """Set and get should work correctly."""
        store = InMemoryCacheStore()
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
        """Expired entries should return None."""
        store = InMemoryCacheStore(ttl=1)  # 1 second TTL
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

    def test_invalidate_removes_entry(self):
        """Invalidate should remove a specific cache entry."""
        store = InMemoryCacheStore()
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media={'key': 'value'},
            data=None,
            text=None,
            timestamp=time.time(),
        )

        store.set('key1', cached)
        store.set('key2', cached)

        store.invalidate('key1')

        assert store.get('key1') is None
        assert store.get('key2') is not None

    def test_invalidate_prefix_removes_matching_entries(self):
        """Invalidate prefix should remove all entries with matching prefix."""
        store = InMemoryCacheStore()
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media={'key': 'value'},
            data=None,
            text=None,
            timestamp=time.time(),
        )

        store.set('/items', cached)
        store.set('/items?page=1', cached)
        store.set('/items?page=2', cached)
        store.set('/other', cached)

        store.invalidate_prefix('/items')

        assert store.get('/items') is None
        assert store.get('/items?page=1') is None
        assert store.get('/items?page=2') is None
        assert store.get('/other') is not None

    def test_cleanup_on_threshold(self):
        """Store should clean up expired entries when threshold is reached."""
        store = InMemoryCacheStore(ttl=1)

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
        """Async get and set should work correctly."""
        store = InMemoryCacheStore()
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

    @pytest.mark.asyncio
    async def test_async_invalidate(self):
        """Async invalidate should work correctly."""
        store = InMemoryCacheStore()
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media={'key': 'value'},
            data=None,
            text=None,
            timestamp=time.time(),
        )

        await store.set_async('key1', cached)
        await store.invalidate_async('key1')

        result = await store.get_async('key1')
        assert result is None

    @pytest.mark.asyncio
    async def test_async_invalidate_prefix(self):
        """Async invalidate prefix should work correctly."""
        store = InMemoryCacheStore()
        cached = CachedResponse(
            status='200 OK',
            headers={},
            media={'key': 'value'},
            data=None,
            text=None,
            timestamp=time.time(),
        )

        await store.set_async('/items', cached)
        await store.set_async('/items?page=1', cached)
        await store.set_async('/other', cached)

        await store.invalidate_prefix_async('/items')

        assert await store.get_async('/items') is None
        assert await store.get_async('/items?page=1') is None
        assert await store.get_async('/other') is not None


class TestCacheMiddlewareIntegration:
    """Integration tests for CacheMiddleware with custom stores."""

    def test_custom_store(self, asgi, util):
        """Test using a custom store implementation."""

        class CustomStore:
            def __init__(self):
                self._data = {}

            def get(self, key):
                return self._data.get(key)

            def set(self, key, response, ttl=None):
                self._data[key] = response

            def invalidate(self, key):
                if key in self._data:
                    del self._data[key]

            def invalidate_prefix(self, prefix):
                keys_to_remove = [k for k in self._data if k.startswith(prefix)]
                for k in keys_to_remove:
                    del self._data[k]

            async def get_async(self, key):
                return self.get(key)

            async def set_async(self, key, response, ttl=None):
                self.set(key, response, ttl)

            async def invalidate_async(self, key):
                self.invalidate(key)

            async def invalidate_prefix_async(self, prefix):
                self.invalidate_prefix(prefix)

        custom_store = CustomStore()
        middleware = CacheMiddleware(store=custom_store)
        app = util.create_app(asgi, middleware=[middleware])
        client = testing.TestClient(app)

        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # First request - cached
        result1 = client.simulate_get('/items')
        # Second request - from cache
        result2 = client.simulate_get('/items')

        assert result1.json['id'] == 1
        assert result2.json['id'] == 1

    def test_middleware_with_cors(self, asgi, util):
        """Test that CacheMiddleware works alongside CORSMiddleware."""
        cache = CacheMiddleware()
        cors = falcon.CORSMiddleware()
        app = util.create_app(asgi, middleware=[cache, cors])
        client = testing.TestClient(app)

        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        result1 = client.simulate_get(
            '/items',
            headers={'Origin': 'http://example.com'},
        )
        result2 = client.simulate_get(
            '/items',
            headers={'Origin': 'http://example.com'},
        )

        assert result1.json['id'] == 1
        assert result2.json['id'] == 1
        assert result1.headers.get('Access-Control-Allow-Origin') == '*'


class TestCacheKeyFormat:
    """Tests for cache key format."""

    def test_cache_key_without_query_string(self):
        """Cache key should not include '?' when there's no query string."""
        middleware = CacheMiddleware()

        import falcon.testing

        req = falcon.testing.create_req(method='GET', path='/items')
        cache_key = middleware._create_cache_key(req)

        assert cache_key == '/items'
        assert '?' not in cache_key

    def test_cache_key_with_query_string(self):
        """Cache key should include '?' when there's a query string."""
        middleware = CacheMiddleware()

        import falcon.testing

        req = falcon.testing.create_req(
            method='GET', path='/items', query_string='foo=bar'
        )
        cache_key = middleware._create_cache_key(req)

        assert cache_key == '/items?foo=bar'
        assert '?' in cache_key

    def test_cache_key_normalizes_query_parameter_order_simple(self):
        """Cache key should normalize query parameters so different order produces same key."""
        middleware = CacheMiddleware()

        import falcon.testing

        # Two requests with same params but different order
        req1 = falcon.testing.create_req(
            method='GET', path='/items', query_string='foo=1&bar=2'
        )
        req2 = falcon.testing.create_req(
            method='GET', path='/items', query_string='bar=2&foo=1'
        )

        cache_key1 = middleware._create_cache_key(req1)
        cache_key2 = middleware._create_cache_key(req2)

        # Both should produce the same cache key
        assert cache_key1 == cache_key2
        # Parameters should be sorted alphabetically
        assert cache_key1 == '/items?bar=2&foo=1'

    def test_cache_key_normalizes_many_parameters_alphabetically(self):
        """Cache key should sort many parameters in true alphabetical order."""
        middleware = CacheMiddleware()

        import falcon.testing

        # Test with 6 parameters in various orders to prove true alphabetical sorting
        # Expected alphabetical order: alpha, beta, delta, gamma, omega, zeta
        expected_sorted = 'alpha=1&beta=2&delta=4&gamma=3&omega=5&zeta=6'

        # Test multiple different orderings
        test_orderings = [
            'zeta=6&omega=5&gamma=3&delta=4&beta=2&alpha=1',  # reverse order
            'gamma=3&alpha=1&zeta=6&beta=2&omega=5&delta=4',  # random order 1
            'delta=4&zeta=6&alpha=1&omega=5&beta=2&gamma=3',  # random order 2
            'beta=2&gamma=3&alpha=1&delta=4&zeta=6&omega=5',  # random order 3
            'omega=5&delta=4&gamma=3&zeta=6&alpha=1&beta=2',  # random order 4
            expected_sorted,  # already sorted
        ]

        cache_keys = []
        for query_string in test_orderings:
            req = falcon.testing.create_req(
                method='GET', path='/items', query_string=query_string
            )
            cache_key = middleware._create_cache_key(req)
            cache_keys.append(cache_key)

        # All orderings should produce the same cache key
        expected_key = f'/items?{expected_sorted}'
        for i, cache_key in enumerate(cache_keys):
            assert cache_key == expected_key, (
                f'Ordering {i} produced {cache_key}, expected {expected_key}'
            )

    def test_cache_key_normalizes_parameters_with_similar_prefixes(self):
        """Cache key should correctly sort parameters with similar prefixes."""
        middleware = CacheMiddleware()

        import falcon.testing

        # Parameters with similar prefixes that could trip up naive sorting
        # Expected order: a, aa, aaa, ab, abc, b, ba
        expected_sorted = 'a=1&aa=2&aaa=3&ab=4&abc=5&b=6&ba=7'

        test_orderings = [
            'ba=7&abc=5&aaa=3&b=6&aa=2&ab=4&a=1',
            'aaa=3&a=1&abc=5&ba=7&aa=2&b=6&ab=4',
            'b=6&a=1&ba=7&aa=2&ab=4&aaa=3&abc=5',
        ]

        for query_string in test_orderings:
            req = falcon.testing.create_req(
                method='GET', path='/items', query_string=query_string
            )
            cache_key = middleware._create_cache_key(req)
            assert cache_key == f'/items?{expected_sorted}', (
                f'Query {query_string} produced {cache_key}'
            )

    def test_cache_key_normalizes_numeric_parameter_names(self):
        """Cache key should correctly sort numeric parameter names."""
        middleware = CacheMiddleware()

        import falcon.testing

        # Numeric strings should sort lexicographically, not numerically
        # "10" comes before "2" in lexicographic order
        expected_sorted = '1=a&10=b&100=c&2=d&20=e&3=f'

        test_orderings = [
            '3=f&20=e&2=d&100=c&10=b&1=a',
            '100=c&2=d&1=a&20=e&3=f&10=b',
            '2=d&3=f&1=a&10=b&20=e&100=c',
        ]

        for query_string in test_orderings:
            req = falcon.testing.create_req(
                method='GET', path='/items', query_string=query_string
            )
            cache_key = middleware._create_cache_key(req)
            assert cache_key == f'/items?{expected_sorted}', (
                f'Query {query_string} produced {cache_key}'
            )

    def test_cache_key_normalizes_multiple_values_for_same_key(self):
        """Cache key should handle multiple values for same parameter."""
        middleware = CacheMiddleware()

        import falcon.testing

        # Requests with multiple values for same parameter in different order
        req1 = falcon.testing.create_req(
            method='GET', path='/items', query_string='tag=a&tag=b&tag=c'
        )
        req2 = falcon.testing.create_req(
            method='GET', path='/items', query_string='tag=b&tag=c&tag=a'
        )

        cache_key1 = middleware._create_cache_key(req1)
        cache_key2 = middleware._create_cache_key(req2)

        # Both should produce the same cache key after sorting
        assert cache_key1 == cache_key2

    def test_cache_key_normalizes_mixed_params_and_repeated_values(self):
        """Cache key should handle mix of different params and repeated values."""
        middleware = CacheMiddleware()

        import falcon.testing

        # Complex case: multiple parameters, some repeated
        # After sorting by (key, value): color=blue, color=red, page=1, size=10, sort=asc
        expected_sorted = 'color=blue&color=red&page=1&size=10&sort=asc'

        test_orderings = [
            'sort=asc&color=red&page=1&color=blue&size=10',
            'page=1&size=10&sort=asc&color=blue&color=red',
            'color=blue&sort=asc&color=red&size=10&page=1',
            'size=10&color=red&sort=asc&page=1&color=blue',
        ]

        for query_string in test_orderings:
            req = falcon.testing.create_req(
                method='GET', path='/items', query_string=query_string
            )
            cache_key = middleware._create_cache_key(req)
            assert cache_key == f'/items?{expected_sorted}', (
                f'Query {query_string} produced {cache_key}'
            )

    def test_cache_key_normalizes_special_characters_in_values(self):
        """Cache key should handle special characters in parameter values."""
        middleware = CacheMiddleware()

        import falcon.testing

        # Parameters with special characters (URL encoded)
        req1 = falcon.testing.create_req(
            method='GET', path='/items', query_string='z=hello%20world&a=foo%26bar'
        )
        req2 = falcon.testing.create_req(
            method='GET', path='/items', query_string='a=foo%26bar&z=hello%20world'
        )

        cache_key1 = middleware._create_cache_key(req1)
        cache_key2 = middleware._create_cache_key(req2)

        # Both should produce the same cache key
        assert cache_key1 == cache_key2
        # 'a' should come before 'z'
        assert cache_key1.startswith('/items?a=')

    def test_cache_key_normalizes_empty_values(self):
        """Cache key should handle parameters with empty values."""
        middleware = CacheMiddleware()

        import falcon.testing

        req1 = falcon.testing.create_req(
            method='GET', path='/items', query_string='z=&a=&m='
        )
        req2 = falcon.testing.create_req(
            method='GET', path='/items', query_string='m=&z=&a='
        )

        cache_key1 = middleware._create_cache_key(req1)
        cache_key2 = middleware._create_cache_key(req2)

        assert cache_key1 == cache_key2
        assert cache_key1 == '/items?a=&m=&z='


class TestQueryParameterNormalization:
    """Tests for query parameter normalization in caching."""

    def test_same_params_different_order_use_same_cache(self, make_client, asgi):
        """Requests with same params in different order should use same cache entry."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # First request with foo=1&bar=2
        result1 = client.simulate_get('/items', params={'foo': '1', 'bar': '2'})
        assert result1.json['id'] == 1

        # Second request with bar=2&foo=1 (same params, different order)
        # Note: params dict order may vary, so we use query_string directly
        result2 = client.simulate_get('/items', query_string='bar=2&foo=1')
        assert result2.json['id'] == 1  # Should return cached response

        # Verify resource was only called once
        count = GetResourceAsync.call_count if asgi else GetResource.call_count
        assert count == 1

    def test_many_params_different_orders_use_same_cache(self, make_client, asgi):
        """Requests with many params in various orders should all use same cache entry."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # Test with 6 parameters in multiple different orders
        # All should resolve to the same cache entry
        query_orderings = [
            'alpha=1&beta=2&gamma=3&delta=4&omega=5&zeta=6',  # order 1
            'zeta=6&omega=5&delta=4&gamma=3&beta=2&alpha=1',  # reverse
            'gamma=3&alpha=1&zeta=6&beta=2&omega=5&delta=4',  # random 1
            'delta=4&zeta=6&alpha=1&omega=5&beta=2&gamma=3',  # random 2
            'omega=5&beta=2&gamma=3&alpha=1&delta=4&zeta=6',  # random 3
        ]

        # First request should execute the resource
        result1 = client.simulate_get('/items', query_string=query_orderings[0])
        assert result1.json['id'] == 1

        # All other orderings should return the cached response
        for i, query_string in enumerate(query_orderings[1:], start=2):
            result = client.simulate_get('/items', query_string=query_string)
            assert result.json['id'] == 1, (
                f'Query ordering {i} ({query_string}) returned id={result.json["id"]}, '
                f'expected id=1 (cached)'
            )

        # Verify resource was only called once despite 5 requests
        count = GetResourceAsync.call_count if asgi else GetResource.call_count
        assert count == 1, f'Resource was called {count} times, expected 1'

    def test_different_param_values_use_different_cache(self, make_client, asgi):
        """Requests with different param values should use different cache entries."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)

        # First request
        result1 = client.simulate_get('/items', params={'foo': '1'})
        assert result1.json['id'] == 1

        # Second request with different value
        result2 = client.simulate_get('/items', params={'foo': '2'})
        assert result2.json['id'] == 2  # Should be a new response

        # Verify both requests were processed
        count = GetResourceAsync.call_count if asgi else GetResource.call_count
        assert count == 2

    def test_complex_real_world_query_normalization(self, make_client, asgi):
        """Test realistic API query parameters with pagination, filtering, sorting."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/products', resource)

        # Realistic e-commerce API query with various orderings
        # Same logical query: get products, page 2, 20 per page, category=electronics,
        # sort by price ascending, filter by in_stock=true
        query_orderings = [
            'page=2&per_page=20&category=electronics&sort=price&order=asc&in_stock=true',
            'in_stock=true&order=asc&sort=price&category=electronics&per_page=20&page=2',
            'sort=price&page=2&in_stock=true&per_page=20&order=asc&category=electronics',
            'category=electronics&in_stock=true&order=asc&page=2&per_page=20&sort=price',
            'per_page=20&sort=price&order=asc&in_stock=true&category=electronics&page=2',
        ]

        # First request
        result1 = client.simulate_get('/products', query_string=query_orderings[0])
        assert result1.json['id'] == 1

        # All variations should hit cache
        for query_string in query_orderings[1:]:
            result = client.simulate_get('/products', query_string=query_string)
            assert result.json['id'] == 1, f'Query {query_string} missed cache'

        # Verify only one actual resource call
        count = GetResourceAsync.call_count if asgi else GetResource.call_count
        assert count == 1


class TestStatusNormalization:
    """Tests for status normalization."""

    def test_integer_status_normalized_to_string(self, make_client, asgi):
        """Integer status should be normalized to string format."""

        class IntStatusResource:
            def on_get(self, req, resp):
                resp.status = 200  # Integer status
                resp.media = {'success': True}

        class IntStatusResourceAsync:
            async def on_get(self, req, resp):
                resp.status = 200  # Integer status
                resp.media = {'success': True}

        client = make_client()
        resource = IntStatusResourceAsync() if asgi else IntStatusResource()
        client.app.add_route('/items', resource)

        result1 = client.simulate_get('/items')
        result2 = client.simulate_get('/items')

        # Both should have string status
        assert '200' in result1.status
        assert '200' in result2.status
        assert 'OK' in result1.status
        assert 'OK' in result2.status


class TestCacheInvalidationDoesNotAffectOtherPaths:
    """Tests to ensure cache invalidation is path-specific."""

    def test_post_to_one_path_does_not_invalidate_other_paths(self, make_client, asgi):
        """POST to /items should not invalidate cache for /other."""
        client = make_client()
        resource = GetResourceAsync() if asgi else GetResource()
        client.app.add_route('/items', resource)
        client.app.add_route('/other', resource)

        # Cache both paths
        client.simulate_get('/items')  # id=1
        client.simulate_get('/other')  # id=2

        # POST to /items
        client.simulate_post('/items')  # id=3

        # /items should be invalidated
        result_items = client.simulate_get('/items')
        assert result_items.json['id'] == 4

        # /other should still be cached
        result_other = client.simulate_get('/other')
        assert result_other.json['id'] == 2


class TestConcurrentRequests:
    """Tests for thread safety."""

    def test_concurrent_get_requests_use_cache(self):
        """Concurrent GET requests should properly use the cache."""
        import threading

        class SlowResource:
            call_count = 0
            call_lock = threading.Lock()

            def on_get(self, req, resp):
                with SlowResource.call_lock:
                    SlowResource.call_count += 1
                    current_count = SlowResource.call_count
                # Simulate slow processing
                time.sleep(0.1)
                resp.media = {'id': current_count}

        SlowResource.call_count = 0

        app = falcon.App(middleware=[CacheMiddleware()])
        app.add_route('/items', SlowResource())

        # First request to populate cache
        client = testing.TestClient(app)
        first_result = client.simulate_get('/items')
        assert first_result.json['id'] == 1

        # Concurrent requests should all get cached response
        results = []
        errors = []

        def make_request():
            try:
                c = testing.TestClient(app)
                result = c.simulate_get('/items')
                results.append(result.json)
            except Exception as e:
                errors.append(str(e))

        threads = []
        for _ in range(10):
            t = threading.Thread(target=make_request)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f'Errors occurred: {errors}'
        assert len(results) == 10
        # All should have id=1 (from cache)
        assert all(r['id'] == 1 for r in results)
        # Handler should only be called once
        assert SlowResource.call_count == 1
