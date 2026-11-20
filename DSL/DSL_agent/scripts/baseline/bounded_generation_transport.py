"""One exact public addendum through the original Config/Provider/HTTP worker.

The original Provider produces its original body. This explicit wire adapter
validates that body using the original approval validator, then supplies only
the independently frozen augmented payload to the unchanged HTTP post method.
"""
from dataclasses import asdict
from pathlib import Path

from candidate_contract import strict_json
from deepseek_provider import HTTPSTransport, DeepSeekProvider
from generation_protocol import Config, check
from delivery_component import bound, read_bytes, require, sha
import bounded_generation_context as context

VERSION = 'poseidon-bounded-single-output-wire-adapter-v1'


def source_manifest():
    import deepseek_http_worker
    import deepseek_provider
    import generation_protocol
    paths = [Path(__file__).absolute(), *[Path(m.__file__).absolute()
             for m in (deepseek_http_worker, deepseek_provider, generation_protocol)]]
    return dict(context.source_manifest(), **{str(p): sha(read_bytes(p)) for p in paths})


def freeze(selection, request, config):
    require(type(config) is Config, 'Original Provider Config required')
    base, actual = context.bodies(selection, request, config)
    return bound(dict(format=VERSION, request_id=request['request_id'], context_binding=selection['binding'],
        configuration=asdict(config), baseline_body_sha256=sha(base), actual_body_sha256=sha(actual),
        actual_body_bytes=len(actual), source_sha256=source_manifest(), original_HTTP_post_and_worker=True,
        Provider_calls=1, HTTP_attempts=1, retries=0, private_references_included=False))


class ContextHTTPSTransport(HTTPSTransport):
    """No credential loading, endpoint selection, retries or alternate HTTP path."""
    def __init__(self, *, selection, request, approved_body, api_key, enabled=False, approved_config=None):
        config = approved_config or Config(max_calls=1)
        expected = freeze(selection, request, config)
        require(approved_body == expected, 'Exact augmented body must be frozen before creating live transport')
        super().__init__(api_key=api_key, approved_request_id=request['request_id'],
                         enabled=enabled, approved_config=config)
        self.selection = selection
        self.request = request
        self.approved_body = approved_body
        self._baseline, self._actual = context.bodies(selection, request, config)

    def validate(self, body, timeout):
        # The existing request/settings/approval validator remains authoritative.
        super().validate(body, timeout)
        check(body == self._baseline, 'request_not_covered_by_approval')
        try:
            require(self.approved_body == freeze(self.selection, self.request, self._config),
                    'Frozen augmented body/source changed')
            base, actual = context.bodies(self.selection, self.request, self._config)
            require(base == body and actual == self._actual, 'Exact augmented body changed')
        except (ValueError, KeyError, TypeError, OSError):
            from generation_protocol import ProviderError
            raise ProviderError('request_not_covered_by_approval') from None
        return strict_json(actual.decode('utf-8'))


def checked_provider(provider, selection, request, frozen_body, *, test_only=False):
    require(type(provider) is DeepSeekProvider and 'generate' not in vars(provider), 'Unmodified original Provider required')
    require(provider.generation_attempts == provider.request_attempts == 0, 'Fresh single-output Provider required')
    require(frozen_body == freeze(selection, request, provider.config), 'Provider/context/body configuration changed')
    transport = provider._transport
    if test_only:
        require(not isinstance(transport, HTTPSTransport) and provider.kind == 'deepseek_offline', 'Fixture cannot claim live Agent')
    else:
        require(type(transport) is ContextHTTPSTransport and transport.approved_body == frozen_body
                and transport.selection == selection and transport.request == request
                and ContextHTTPSTransport.post is HTTPSTransport.post,
                'Original HTTP worker with explicitly frozen context adapter required')
    return provider
