import re
import uuid

import pytest

import falcon
from falcon import testing
from falcon.middleware import RequestIDMiddleware


class SimpleResource:
    """Resource that returns the request ID from context."""

    def on_get(self, req, resp):
        resp.media = {
            'context_request_id': getattr(req.context, 'request_id', None),
            'header_request_id': req.request_id,
        }

    def on_post(self, req, resp):
        resp.media = {
            'context_request_id': getattr(req.context, 'request_id', None),
            'header_request_id': req.request_id,
        }
        resp.status = falcon.HTTP_201


class SimpleResourceAsync:
    """Async resource that returns the request ID from context."""

    async def on_get(self, req, resp):
        resp.media = {
            'context_request_id': getattr(req.context, 'request_id', None),
            'header_request_id': req.request_id,
        }

    async def on_post(self, req, resp):
        resp.media = {
            'context_request_id': getattr(req.context, 'request_id', None),
            'header_request_id': req.request_id,
        }
        resp.status = falcon.HTTP_201


class ErrorResource:
    """Resource that raises an error."""

    def on_get(self, req, resp):
        raise falcon.HTTPBadRequest(description='Test error')


class ErrorResourceAsync:
    """Async resource that raises an error."""

    async def on_get(self, req, resp):
        raise falcon.HTTPBadRequest(description='Test error')


@pytest.fixture
def make_client(asgi, util):
    """Factory fixture to create test clients with request ID middleware."""

    def _make_client(middleware=None, **middleware_kwargs):
        if middleware is None:
            middleware = RequestIDMiddleware(**middleware_kwargs)
        app = util.create_app(asgi, middleware=[middleware])
        return testing.TestClient(app)

    return _make_client


class TestRequestIDMiddleware:
    """Tests for RequestIDMiddleware."""

    def test_generates_request_id_when_not_provided(self, make_client, asgi):
        """When no X-Request-ID header is sent, middleware generates one."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get('/test')

        assert result.status == falcon.HTTP_200
        # Check that a request ID was generated
        response_id = result.headers.get('X-Request-ID')
        assert response_id is not None
        # Should be a valid UUID
        assert len(response_id) == 36
        uuid.UUID(response_id)  # Validates UUID format

        # Context should have the same ID
        assert result.json['context_request_id'] == response_id
        # req.request_id property should also return the generated ID
        assert result.json['header_request_id'] == response_id

    def test_uses_provided_request_id(self, make_client, asgi):
        """When X-Request-ID header is sent, middleware uses it."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        provided_id = 'my-custom-request-id-123'
        result = client.simulate_get('/test', headers={'X-Request-ID': provided_id})

        assert result.status == falcon.HTTP_200
        # Check that the provided ID is used
        response_id = result.headers.get('X-Request-ID')
        assert response_id == provided_id

        # Context should have the same ID
        assert result.json['context_request_id'] == provided_id
        # Header property should also return the same ID
        assert result.json['header_request_id'] == provided_id

    def test_request_id_in_response_headers(self, make_client, asgi):
        """Request ID should always be present in response headers."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        # Without providing a header
        result1 = client.simulate_get('/test')
        assert 'X-Request-ID' in result1.headers

        # With providing a header
        result2 = client.simulate_get('/test', headers={'X-Request-ID': 'test-id'})
        assert result2.headers.get('X-Request-ID') == 'test-id'

    def test_custom_header_name(self, make_client, asgi):
        """Middleware should use custom header name when configured."""
        client = make_client(header_name='X-Correlation-ID')
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        provided_id = 'correlation-123'
        result = client.simulate_get(
            '/test', headers={'X-Correlation-ID': provided_id}
        )

        assert result.status == falcon.HTTP_200
        # Check custom header name is used in response
        assert result.headers.get('X-Correlation-ID') == provided_id
        # Default header should not be present
        assert result.headers.get('X-Request-ID') is None

        # Context should have the ID
        assert result.json['context_request_id'] == provided_id

    def test_custom_generator(self, make_client, asgi):
        """Middleware should use custom generator when configured."""
        counter = [0]

        def custom_generator():
            counter[0] += 1
            return f'custom-{counter[0]}'

        client = make_client(generator=custom_generator)
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        result1 = client.simulate_get('/test')
        result2 = client.simulate_get('/test')

        assert result1.headers.get('X-Request-ID') == 'custom-1'
        assert result2.headers.get('X-Request-ID') == 'custom-2'

    def test_request_id_on_error(self, make_client, asgi):
        """Request ID should be present in response even on errors."""
        client = make_client()
        resource = ErrorResourceAsync() if asgi else ErrorResource()
        client.app.add_route('/test', resource)

        provided_id = 'error-request-id'
        result = client.simulate_get('/test', headers={'X-Request-ID': provided_id})

        assert result.status == falcon.HTTP_400
        # Request ID should still be in response headers
        assert result.headers.get('X-Request-ID') == provided_id

    def test_works_with_post_requests(self, make_client, asgi):
        """Request ID should work with POST requests."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        provided_id = 'post-request-id'
        result = client.simulate_post(
            '/test',
            headers={'X-Request-ID': provided_id},
            json={'data': 'test'},
        )

        assert result.status == falcon.HTTP_201
        assert result.headers.get('X-Request-ID') == provided_id
        assert result.json['context_request_id'] == provided_id

    def test_each_request_gets_unique_id(self, make_client, asgi):
        """Each request without ID should get a unique generated ID."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        result1 = client.simulate_get('/test')
        result2 = client.simulate_get('/test')
        result3 = client.simulate_get('/test')

        id1 = result1.headers.get('X-Request-ID')
        id2 = result2.headers.get('X-Request-ID')
        id3 = result3.headers.get('X-Request-ID')

        # All IDs should be unique
        assert id1 != id2
        assert id2 != id3
        assert id1 != id3


class TestRequestIDMiddlewareIntegration:
    """Integration tests for RequestIDMiddleware with other middleware."""

    def test_works_with_cors_middleware(self, asgi, util):
        """RequestIDMiddleware should work alongside CORSMiddleware."""
        request_id_middleware = RequestIDMiddleware()
        cors_middleware = falcon.CORSMiddleware()
        app = util.create_app(asgi, middleware=[request_id_middleware, cors_middleware])
        client = testing.TestClient(app)

        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        result = client.simulate_get(
            '/test',
            headers={
                'X-Request-ID': 'cors-test-id',
                'Origin': 'http://example.com',
            },
        )

        assert result.status == falcon.HTTP_200
        assert result.headers.get('X-Request-ID') == 'cors-test-id'
        assert result.headers.get('Access-Control-Allow-Origin') == '*'

    def test_works_with_idempotency_middleware(self, asgi, util):
        """RequestIDMiddleware should work alongside IdempotencyMiddleware."""
        request_id_middleware = RequestIDMiddleware()
        idempotency_middleware = falcon.IdempotencyMiddleware()
        app = util.create_app(
            asgi, middleware=[request_id_middleware, idempotency_middleware]
        )
        client = testing.TestClient(app)

        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        result = client.simulate_post(
            '/test',
            headers={
                'X-Request-ID': 'idempotent-request-id',
                'Idempotency-Key': 'idem-key-123',
            },
            json={'data': 'test'},
        )

        assert result.status == falcon.HTTP_201
        assert result.headers.get('X-Request-ID') == 'idempotent-request-id'


class TestRequestIDProperty:
    """Tests for the request_id property on Request objects."""

    def test_wsgi_request_id_property_from_header(self):
        """WSGI Request.request_id should read from header when context not set."""
        env = testing.create_environ(
            headers={'X-Request-ID': 'wsgi-test-id'}
        )
        req = falcon.Request(env)

        assert req.request_id == 'wsgi-test-id'

    def test_wsgi_request_id_property_from_context(self):
        """WSGI Request.request_id should read from context first."""
        env = testing.create_environ(
            headers={'X-Request-ID': 'header-id'}
        )
        req = falcon.Request(env)
        req.context.request_id = 'context-id'

        # Context should take precedence over header
        assert req.request_id == 'context-id'

    def test_wsgi_request_id_property_context_when_no_header(self):
        """WSGI Request.request_id should return context value when no header."""
        env = testing.create_environ()
        req = falcon.Request(env)
        req.context.request_id = 'generated-id'

        assert req.request_id == 'generated-id'

    def test_wsgi_request_id_property_none_when_missing(self):
        """WSGI Request.request_id should be None when neither context nor header."""
        env = testing.create_environ()
        req = falcon.Request(env)

        assert req.request_id is None

    def test_asgi_request_id_property_from_header(self, asgi):
        """ASGI Request.request_id should read from header when context not set."""
        if not asgi:
            pytest.skip('ASGI-only test')

        import falcon.asgi

        scope = testing.create_scope(headers={'X-Request-ID': 'asgi-test-id'})

        async def receive():
            return {'type': 'http.request', 'body': b''}

        req = falcon.asgi.Request(scope, receive)

        assert req.request_id == 'asgi-test-id'

    def test_asgi_request_id_property_from_context(self, asgi):
        """ASGI Request.request_id should read from context first."""
        if not asgi:
            pytest.skip('ASGI-only test')

        import falcon.asgi

        scope = testing.create_scope(headers={'X-Request-ID': 'header-id'})

        async def receive():
            return {'type': 'http.request', 'body': b''}

        req = falcon.asgi.Request(scope, receive)
        req.context.request_id = 'context-id'

        # Context should take precedence over header
        assert req.request_id == 'context-id'

    def test_asgi_request_id_property_context_when_no_header(self, asgi):
        """ASGI Request.request_id should return context value when no header."""
        if not asgi:
            pytest.skip('ASGI-only test')

        import falcon.asgi

        scope = testing.create_scope()

        async def receive():
            return {'type': 'http.request', 'body': b''}

        req = falcon.asgi.Request(scope, receive)
        req.context.request_id = 'generated-id'

        assert req.request_id == 'generated-id'

    def test_asgi_request_id_property_none_when_missing(self, asgi):
        """ASGI Request.request_id should be None when neither context nor header."""
        if not asgi:
            pytest.skip('ASGI-only test')

        import falcon.asgi

        scope = testing.create_scope()

        async def receive():
            return {'type': 'http.request', 'body': b''}

        req = falcon.asgi.Request(scope, receive)

        assert req.request_id is None


class TestRequestIDEmptyStringHandling:
    """Tests for empty string header handling."""

    def test_wsgi_empty_string_header_returns_none(self):
        """WSGI Request.request_id should return None for empty string header."""
        env = testing.create_environ(headers={'X-Request-ID': ''})
        req = falcon.Request(env)

        # Empty string should be treated as None
        assert req.request_id is None

    def test_asgi_empty_string_header_returns_none(self, asgi):
        """ASGI Request.request_id should return None for empty string header."""
        if not asgi:
            pytest.skip('ASGI-only test')

        import falcon.asgi

        scope = testing.create_scope(headers={'X-Request-ID': ''})

        async def receive():
            return {'type': 'http.request', 'body': b''}

        req = falcon.asgi.Request(scope, receive)

        # Empty string should be treated as None
        assert req.request_id is None

    def test_middleware_generates_id_for_empty_string_header(self, make_client, asgi):
        """Middleware should generate ID when header is empty string."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        # Send empty string header
        result = client.simulate_get('/test', headers={'X-Request-ID': ''})

        assert result.status == falcon.HTTP_200

        # Middleware should have generated a new ID
        response_id = result.headers.get('X-Request-ID')
        assert response_id is not None
        assert response_id != ''
        # Should be a valid UUID
        uuid.UUID(response_id)

        # Both context and property should return the same generated ID
        assert result.json['context_request_id'] == response_id
        assert result.json['header_request_id'] == response_id

    def test_request_id_property_matches_context_with_empty_header(self, make_client, asgi):
        """req.request_id should match req.context.request_id when header was empty."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        # Send empty string header
        result = client.simulate_get('/test', headers={'X-Request-ID': ''})

        # Both should be the same generated ID (not empty string vs generated)
        assert result.json['context_request_id'] == result.json['header_request_id']
        assert result.json['header_request_id'] != ''

    def test_wsgi_empty_context_falls_back_to_header(self):
        """WSGI: Empty string in context should fall back to header."""
        env = testing.create_environ(headers={'X-Request-ID': 'valid-id'})
        req = falcon.Request(env)
        req.context.request_id = ''

        # Empty context should fall back to header
        assert req.request_id == 'valid-id'

    def test_asgi_empty_context_falls_back_to_header(self, asgi):
        """ASGI: Empty string in context should fall back to header."""
        if not asgi:
            pytest.skip('ASGI-only test')

        import falcon.asgi

        scope = testing.create_scope(headers={'X-Request-ID': 'valid-id'})

        async def receive():
            return {'type': 'http.request', 'body': b''}

        req = falcon.asgi.Request(scope, receive)
        req.context.request_id = ''

        # Empty context should fall back to header
        assert req.request_id == 'valid-id'


class TestRequestIDMiddlewareConfiguration:
    """Tests for RequestIDMiddleware configuration options."""

    def test_default_header_name(self):
        """Default header name should be X-Request-ID."""
        middleware = RequestIDMiddleware()
        assert middleware._header_name == 'X-Request-ID'

    def test_custom_header_name_configuration(self):
        """Custom header name should be configurable."""
        middleware = RequestIDMiddleware(header_name='X-Trace-ID')
        assert middleware._header_name == 'X-Trace-ID'

    def test_default_generator_creates_uuids(self):
        """Default generator should create valid UUIDs."""
        middleware = RequestIDMiddleware()
        generated_id = middleware._generator()

        # Should be a valid UUID4 string
        assert len(generated_id) == 36
        parsed = uuid.UUID(generated_id)
        assert parsed.version == 4

    def test_custom_generator_is_used(self):
        """Custom generator should be used when provided."""
        custom_id = 'my-fixed-id'
        middleware = RequestIDMiddleware(generator=lambda: custom_id)

        assert middleware._generator() == custom_id


class TestRequestIDWithDifferentHeaderFormats:
    """Tests for request ID with various header value formats."""

    def test_uuid_format(self, make_client, asgi):
        """Should handle UUID format request IDs."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        uuid_id = str(uuid.uuid4())
        result = client.simulate_get('/test', headers={'X-Request-ID': uuid_id})

        assert result.headers.get('X-Request-ID') == uuid_id

    def test_short_alphanumeric_format(self, make_client, asgi):
        """Should handle short alphanumeric request IDs."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        short_id = 'abc123'
        result = client.simulate_get('/test', headers={'X-Request-ID': short_id})

        assert result.headers.get('X-Request-ID') == short_id

    def test_prefixed_format(self, make_client, asgi):
        """Should handle prefixed request IDs."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        prefixed_id = 'req-123-abc-456'
        result = client.simulate_get('/test', headers={'X-Request-ID': prefixed_id})

        assert result.headers.get('X-Request-ID') == prefixed_id

    def test_long_request_id(self, make_client, asgi):
        """Should handle long request IDs."""
        client = make_client()
        resource = SimpleResourceAsync() if asgi else SimpleResource()
        client.app.add_route('/test', resource)

        long_id = 'x' * 200
        result = client.simulate_get('/test', headers={'X-Request-ID': long_id})

        assert result.headers.get('X-Request-ID') == long_id
