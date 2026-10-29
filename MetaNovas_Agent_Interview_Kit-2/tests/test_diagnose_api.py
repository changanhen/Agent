"""Offline checks that diagnostic output excludes secrets and response content."""

import io
import json
import unittest
import urllib.error
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts import diagnose_api as diagnostic


class DiagnosticTests(unittest.TestCase):
    def test_http_error_is_categorized_without_echoing_body(self):
        error = RuntimeError('Provider returned HTTP 401: SECRET_DO_NOT_PRINT')
        value = diagnostic.error_info(error)
        self.assertEqual(value['category'], 'authentication_failed')
        self.assertNotIn('SECRET_DO_NOT_PRINT', json.dumps(value))

    def test_model_and_payment_errors_are_distinct(self):
        for status, category in ((402, 'balance_or_payment_required'), (404, 'model_or_endpoint_not_found')):
            self.assertEqual(diagnostic.error_info(RuntimeError('Provider returned HTTP ' + str(status)))['category'], category)

    def test_response_metadata_reports_truncation_without_content(self):
        raw = json.dumps({'choices': [{'finish_reason': 'length', 'message': {
            'content': 'PRIVATE_RECORD', 'reasoning_content': 'PRIVATE_REASONING'}}]}).encode()
        stream = io.StringIO()
        with redirect_stdout(stream):
            diagnostic.describe_response(raw)
        value = json.loads(stream.getvalue())
        self.assertEqual(value['finish_reason'], 'length')
        self.assertNotIn('PRIVATE_', stream.getvalue())

    def test_missing_key_stops_before_network(self):
        stream = io.StringIO()
        with patch.dict(diagnostic.os.environ, {}, clear=True), redirect_stdout(stream):
            with patch.object(diagnostic.sdk, 'chat') as chat:
                self.assertEqual(diagnostic.main([]), 2)
                chat.assert_not_called()
        self.assertIn('missing_api_key', stream.getvalue())

    def test_smoke_uses_sdk_and_keeps_key_out_of_output(self):
        raw = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': 'OK'}}]}).encode()
        stream = io.StringIO()
        with patch.dict(diagnostic.os.environ, {'INTERVIEW_PROVIDER': 'deepseek', 'INTERVIEW_API_KEY': 'SECRET_DO_NOT_PRINT'}, clear=True):
            with patch.object(diagnostic.sdk.urllib.request, 'urlopen') as network, redirect_stdout(stream):
                network.return_value.__enter__.return_value.read.return_value = raw
                self.assertEqual(diagnostic.main([]), 0)
        self.assertNotIn('SECRET_DO_NOT_PRINT', stream.getvalue())
        self.assertIn('"usable_output": true', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
