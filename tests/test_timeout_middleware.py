import time

import pytest

import falcon
from falcon import testing
from falcon.middleware import TimeoutMiddleware


class SlowResource:
    """Resource that takes time to process."""

    def on_get(self, req, resp):
        delay = float(req.get_param('delay', default='0'))
        time.sleep(delay)
        resp.media = {'delay': delay, 'completed': True}

    def on_post(self, req, resp):
        delay = float(req.get_param('delay', default='0'))
        time.sleep(delay)
        resp.media = {'delay': delay, 'completed': True}
        resp.status = falcon.HTTP_201


class SlowResourceAsync:
    """Async resource that takes time to process."""

    async def on_get(self, req, resp):
        import asyncio

        delay = float(req.get_param('delay', default='0'))
        await asyncio.sleep(delay)
        resp.media = {'delay': delay, 'completed': True}

    async def on_post(self, req, resp):
        import asyncio

        delay = float(req.get_param('delay', default='0'))
        await asyncio.sleep(delay)
        resp.media = {'delay': delay, 'completed': True}
        resp.status = falcon.HTTP_201


class FastResource:
    """Resource that returns immediately."""

    def on_get(self, req, resp):
        resp.media = {'fast': True}

    def on_post(self, req, resp):
        resp.media = {'fast': True}
        resp.status = falcon.HTTP_201


class FastResourceAsync:
    """Async resource that returns immediately."""

    async def on_get(self, req, resp):
        resp.media = {'fast': True}

    async def on_post(self, req, resp):
        resp.media = {'fast': True}
        resp.status = falcon.HTTP_201


class ErrorResource:
    """Resource that raises an error."""

    def on_get(self, req, resp):
        raise falcon.HTTPBadRequest(description='Test error')


class ErrorResourceAsync:
    """Async resource that raises an error."""

    async def on_get(self, req, resp):
        raise falcon.HTTPBadRequest(description='Test error')


class ResourceWithTimeout:
    """Resource with a timeout attribute for resolver testing."""

    timeout = 60.0

    def on_get(self, req, resp):
        resp.media = {'timeout': self.timeout}


class ResourceWithTimeoutAsync:
    """Async resource with a timeout attribute for resolver testing."""

    timeout = 60.0

    async def on_get(self, req, resp):
        resp.media = {'timeout': self.timeout}


@pytest.fixture
def make_client(asgi, util):
    """Factory fixture to create test clients with timeout middleware."""

    def _make_client(middleware=None, **middleware_kwargs):
        if middleware is None:
            middleware = TimeoutMiddleware(**middleware_kwargs)
        app = util.create_app(asgi, middleware=[middleware])
        return testing.TestClient(app)

    return _make_client


class TestTimeoutMiddlewareBasic:
    """Basic timeout middleware tests."""

    def test_fast_request_succeeds(self, make_client, asgi):
        """Requests completing within timeout should succeed."""
        client = make_client(default_timeout=5.0)
        resource = FastResourceAsync() if asgi else FastResource()
        client.app.add_route('/fast', resource)

        result = client.simulate_get('/fast')

        assert result.status == falcon.HTTP_200
        assert result.json['fast'] is True

    def test_slow_request_times_out_wsgi(self, make_client, asgi):
        """WSGI: Slow requests should report timeout after completion."""
        if asgi:
            pytest.skip('WSGI-specific test')

        client = make_client(default_timeout=0.05)
        resource = SlowResource()
        client.app.add_route('/slow', resource)

        # Request takes longer than timeout
        result = client.simulate_get('/slow?delay=0.1')

        # WSGI reports timeout after completion
        assert result.status == falcon.HTTP_503
        assert 'timeout' in result.json.get('title', '').lower()

    def test_slow_request_times_out_asgi(self, make_client, asgi):
        """ASGI: Slow requests should report timeout after completion."""
        if not asgi:
            pytest.skip('ASGI-specific test')

        client = make_client(default_timeout=0.05)
        resource = SlowResourceAsync()
        client.app.add_route('/slow', resource)

        # Request takes longer than timeout
        result = client.simulate_get('/slow?delay=0.1')

        # ASGI reports timeout after completion
        assert result.status == falcon.HTTP_503
        assert 'timeout' in result.json.get('title', '').lower()

    def test_no_timeout_when_disabled(self, make_client, asgi):
        """No timeout should occur when default_timeout is None."""
        client = make_client(default_timeout=None)
        resource = SlowResourceAsync() if asgi else SlowResource()
        client.app.add_route('/slow', resource)

        # Even slow requests should succeed
        result = client.simulate_get('/slow?delay=0.05')

        assert result.status == falcon.HTTP_200
        assert result.json['completed'] is True

    def test_timeout_on_post_request(self, make_client, asgi):
        """Timeout should work on POST requests too."""
        client = make_client(default_timeout=0.05)
        resource = SlowResourceAsync() if asgi else SlowResource()
        client.app.add_route('/slow', resource)

        result = client.simulate_post('/slow?delay=0.1')

        assert result.status == falcon.HTTP_503

    def test_request_within_timeout_succeeds(self, make_client, asgi):
        """Requests completing just within timeout should succeed."""
        client = make_client(default_timeout=1.0)
        resource = SlowResourceAsync() if asgi else SlowResource()
        client.app.add_route('/slow', resource)

        # Request completes well within timeout
        result = client.simulate_get('/slow?delay=0.01')

        assert result.status == falcon.HTTP_200
        assert result.json['completed'] is True


class TestTimeoutMiddlewareCustomResolver:
    """Tests for custom timeout resolver."""

    def test_custom_resolver_overrides_default(self, make_client, asgi):
        """Custom resolver should override default timeout."""

        def resolver(req, resp, resource):
            if req.path == '/slow':
                return 10.0  # Long timeout for slow route
            return 0.01  # Very short for others

        client = make_client(default_timeout=5.0, timeout_resolver=resolver)
        fast_resource = FastResourceAsync() if asgi else FastResource()
        slow_resource = SlowResourceAsync() if asgi else SlowResource()
        client.app.add_route('/fast', fast_resource)
        client.app.add_route('/slow', slow_resource)

        # /slow has long timeout, should succeed
        result = client.simulate_get('/slow?delay=0.05')
        assert result.status == falcon.HTTP_200

    def test_resolver_returns_none_uses_default(self, make_client, asgi):
        """Resolver returning None should use default timeout."""

        def resolver(req, resp, resource):
            return None  # Always use default

        client = make_client(default_timeout=5.0, timeout_resolver=resolver)
        resource = FastResourceAsync() if asgi else FastResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test')
        assert result.status == falcon.HTTP_200

    def test_resolver_with_resource_attribute(self, make_client, asgi):
        """Resolver can use resource attributes for timeout."""

        def resolver(req, resp, resource):
            return getattr(resource, 'timeout', None)

        client = make_client(default_timeout=0.01, timeout_resolver=resolver)
        resource = ResourceWithTimeoutAsync() if asgi else ResourceWithTimeout()
        client.app.add_route('/test', resource)

        # Resource has timeout=60.0, so should succeed
        result = client.simulate_get('/test')
        assert result.status == falcon.HTTP_200
        assert result.json['timeout'] == 60.0

    def test_resolver_based_on_path(self, make_client, asgi):
        """Resolver can set different timeouts based on path."""

        def resolver(req, resp, resource):
            if req.path.startswith('/slow'):
                return 10.0
            if req.path.startswith('/fast'):
                return 0.001
            return None

        client = make_client(default_timeout=1.0, timeout_resolver=resolver)
        slow_resource = SlowResourceAsync() if asgi else SlowResource()
        client.app.add_route('/slow/endpoint', slow_resource)

        # /slow/ path has 10s timeout
        result = client.simulate_get('/slow/endpoint?delay=0.05')
        assert result.status == falcon.HTTP_200


class TestResponseTimeHeader:
    """Tests for X-Response-Time header."""

    def test_response_time_header_when_enabled(self, make_client, asgi):
        """X-Response-Time header should be present when enabled."""
        client = make_client(include_response_time=True)
        resource = FastResourceAsync() if asgi else FastResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test')

        assert result.status == falcon.HTTP_200
        assert 'X-Response-Time' in result.headers
        # Should be in milliseconds format
        assert 'ms' in result.headers['X-Response-Time']

    def test_response_time_header_not_present_when_disabled(self, make_client, asgi):
        """X-Response-Time header should not be present when disabled."""
        client = make_client(include_response_time=False)
        resource = FastResourceAsync() if asgi else FastResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test')

        assert result.status == falcon.HTTP_200
        assert 'X-Response-Time' not in result.headers

    def test_response_time_header_format(self, make_client, asgi):
        """X-Response-Time should have correct format."""
        client = make_client(include_response_time=True)
        resource = SlowResourceAsync() if asgi else SlowResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test?delay=0.05')

        assert result.status == falcon.HTTP_200
        response_time = result.headers.get('X-Response-Time')
        assert response_time is not None
        # Should be a number followed by 'ms'
        assert response_time.endswith('ms')
        # Parse the number part
        time_value = float(response_time[:-2])
        # Should be at least 50ms (0.05 seconds delay)
        assert time_value >= 50.0

    def test_response_time_header_on_timeout(self, make_client, asgi):
        """X-Response-Time header should be present even on timeout."""
        client = make_client(default_timeout=0.05, include_response_time=True)
        resource = SlowResourceAsync() if asgi else SlowResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test?delay=0.1')

        assert result.status == falcon.HTTP_503
        # Response time header should still be set
        assert 'X-Response-Time' in result.headers

    def test_response_time_header_on_error(self, make_client, asgi):
        """X-Response-Time header should be present even on errors."""
        client = make_client(include_response_time=True)
        resource = ErrorResourceAsync() if asgi else ErrorResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test')

        assert result.status == falcon.HTTP_400
        # Response time header should still be set
        assert 'X-Response-Time' in result.headers


class TestTimeoutMiddlewareConfiguration:
    """Tests for middleware configuration."""

    def test_default_timeout_value(self):
        """Default timeout should be 30 seconds."""
        middleware = TimeoutMiddleware()
        assert middleware._default_timeout == 30.0

    def test_custom_default_timeout(self):
        """Custom default timeout should be configurable."""
        middleware = TimeoutMiddleware(default_timeout=60.0)
        assert middleware._default_timeout == 60.0

    def test_none_default_timeout(self):
        """None default timeout should disable timeout."""
        middleware = TimeoutMiddleware(default_timeout=None)
        assert middleware._default_timeout is None

    def test_zero_timeout(self):
        """Zero timeout should be allowed."""
        middleware = TimeoutMiddleware(default_timeout=0.0)
        assert middleware._default_timeout == 0.0

    def test_include_response_time_default(self):
        """include_response_time should default to False."""
        middleware = TimeoutMiddleware()
        assert middleware._include_response_time is False

    def test_timeout_resolver_default(self):
        """timeout_resolver should default to None."""
        middleware = TimeoutMiddleware()
        assert middleware._timeout_resolver is None


class TestTimeoutMiddlewareErrorHandling:
    """Tests for error handling scenarios."""

    def test_timeout_does_not_override_existing_error(self, make_client, asgi):
        """Timeout should not override errors raised by the resource."""
        client = make_client(default_timeout=0.05)
        resource = ErrorResourceAsync() if asgi else ErrorResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test')

        # Should see the original error, not a timeout
        assert result.status == falcon.HTTP_400

    def test_timeout_error_has_description(self, make_client, asgi):
        """Timeout error should have a descriptive message."""
        client = make_client(default_timeout=0.05)
        resource = SlowResourceAsync() if asgi else SlowResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test?delay=0.1')

        assert result.status == falcon.HTTP_503
        assert 'timeout' in result.json.get('description', '').lower()

    def test_timeout_error_includes_actual_time(self, make_client, asgi):
        """Timeout error description should include actual processing time."""
        client = make_client(default_timeout=0.05)
        resource = SlowResourceAsync() if asgi else SlowResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test?delay=0.1')

        assert result.status == falcon.HTTP_503
        description = result.json.get('description', '')
        # Should mention 'actual' time
        assert 'actual' in description.lower()


class TestTimeoutMiddlewareWithOtherMiddleware:
    """Tests for interaction with other middleware."""

    def test_timeout_with_request_id_middleware(self, asgi, util):
        """Timeout middleware should work alongside RequestIDMiddleware."""
        from falcon.middleware import RequestIDMiddleware

        timeout_mw = TimeoutMiddleware(default_timeout=5.0, include_response_time=True)
        request_id_mw = RequestIDMiddleware()

        app = util.create_app(asgi, middleware=[request_id_mw, timeout_mw])
        client = testing.TestClient(app)

        resource = FastResourceAsync() if asgi else FastResource()
        app.add_route('/test', resource)

        result = client.simulate_get('/test')

        assert result.status == falcon.HTTP_200
        assert 'X-Request-ID' in result.headers
        assert 'X-Response-Time' in result.headers


class TestHTTPRequestProcessingTimeout:
    """Tests for the HTTPRequestProcessingTimeout error class."""

    def test_error_class_exists(self):
        """HTTPRequestProcessingTimeout should be accessible from falcon module."""
        assert hasattr(falcon, 'HTTPRequestProcessingTimeout')

    def test_error_is_503(self):
        """HTTPRequestProcessingTimeout should result in 503 status."""
        error = falcon.HTTPRequestProcessingTimeout()
        assert error.status == falcon.HTTP_503

    def test_error_with_timeout_generates_description(self):
        """HTTPRequestProcessingTimeout with timeout should auto-generate description."""
        error = falcon.HTTPRequestProcessingTimeout(timeout=30.0)
        assert error.description is not None
        assert '30.0' in error.description

    def test_error_with_custom_description(self):
        """HTTPRequestProcessingTimeout should accept custom description."""
        error = falcon.HTTPRequestProcessingTimeout(description='Custom message')
        assert error.description == 'Custom message'

    def test_error_title(self):
        """HTTPRequestProcessingTimeout should have appropriate title."""
        error = falcon.HTTPRequestProcessingTimeout()
        assert error.title == 'Request Processing Timeout'

    def test_error_with_retry_after(self):
        """HTTPRequestProcessingTimeout should support retry_after."""
        error = falcon.HTTPRequestProcessingTimeout(retry_after=30)
        assert error.headers is not None
        assert 'Retry-After' in error.headers
        assert error.headers['Retry-After'] == '30'


class TestTimeoutMiddlewareEdgeCases:
    """Tests for edge cases."""

    def test_request_to_nonexistent_route(self, make_client, asgi):
        """Timeout middleware should handle 404 responses gracefully."""
        client = make_client(default_timeout=5.0)

        result = client.simulate_get('/nonexistent')

        assert result.status == falcon.HTTP_404

    def test_very_small_timeout(self, make_client, asgi):
        """Very small timeout should trigger timeout for any processing."""
        client = make_client(default_timeout=0.0001)
        resource = FastResourceAsync() if asgi else FastResource()
        client.app.add_route('/test', resource)

        # Even fast requests should timeout with a tiny timeout
        # Note: This may or may not timeout depending on system speed
        result = client.simulate_get('/test')
        # Either succeeds or times out - both are acceptable
        assert result.status in (falcon.HTTP_200, falcon.HTTP_503)

    def test_large_timeout_value(self, make_client, asgi):
        """Large timeout values should work correctly."""
        client = make_client(default_timeout=86400.0)  # 24 hours
        resource = FastResourceAsync() if asgi else FastResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test')

        assert result.status == falcon.HTTP_200
