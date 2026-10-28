import json
import signal
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch
import augment


class Tests(unittest.TestCase):
    def test_cleanup_and_original_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / 'train.jsonl'
            original.write_bytes(b'x' * 1000)
            store = augment.Store(root, 10000, 3000, 1000, 0)
            try:
                for i in range(8):
                    (store.directory / f'aug-{i:032x}.jsonl').write_bytes(b'x' * 1000)
                before = augment.tree_bytes(root)
                with patch('augment.random.shuffle') as shuffle:
                    store.ensure_space(1500)
                    shuffle.assert_called_once()
                self.assertEqual(before - augment.tree_bytes(root), 3000)
                self.assertLessEqual(augment.tree_bytes(root) + 1500, 10000)
                self.assertEqual(original.read_bytes(), b'x' * 1000)
                with self.assertRaises(RuntimeError):
                    augment.Store(root, 10000, 3000, 1000, 0)
            finally:
                store.close()

    def test_insufficient_owned_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / 'original'
            original.write_bytes(b'x' * 11000)
            store = augment.Store(tmp, 10000, 3000, 1000, 0)
            try:
                with self.assertRaises(RuntimeError):
                    store.ensure_space(1)
                self.assertEqual(original.stat().st_size, 11000)
            finally:
                store.close()

    def test_both_api_protocols(self):
        paths = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                paths.append(self.path)
                content = json.dumps({'texts': ['동일 의미의 표현']})
                result = ({'message': {'content': content}} if self.path == '/api/chat'
                          else {'choices': [{'message': {'content': content}}]})
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for backend in ('ollama', 'vllm'):
                args = augment.parser().parse_args(['--backend', backend, '--model', 'test'])
                args.base_url = f'http://127.0.0.1:{server.server_port}'
                self.assertEqual(augment.generate(args, '원문', 1, 5), ['동일 의미의 표현'])
            self.assertEqual(paths, ['/api/chat', '/v1/chat/completions'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    @unittest.skipUnless(hasattr(signal, "SIGALRM"), "Linux SIGALRM test")
    def test_hard_deadline(self):
        args = augment.parser().parse_args(['--model', 'test'])
        args.base_url = 'http://unused'
        with patch('augment.api_json', side_effect=lambda *a: time.sleep(2)):
            with self.assertRaises(augment.Deadline):
                augment.generate(args, '원문', 1, 0.05)

    def test_mock_run_restart_and_split_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'train.jsonl'
            source.write_text('{"text":"숲 복원 사업", "label":1}\n', encoding='utf-8')
            root = Path(tmp) / 'out'
            args = augment.parser().parse_args([
                '--input', str(source), '--output', str(root), '--mock',
                '--max-batches', '2', '--interval', '0', '--until', '2099-01-01T00:00:00+09:00'])
            augment.run(args)
            augment.run(args)
            rows = [json.loads(line) for path in root.rglob('aug-*.jsonl')
                    for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(r['label'] == 1 and r['backend'] == 'mock' for r in rows))
            source.write_text('{"text":"검증", "label":1, "split":"test"}\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                augment.run(args)

    def test_expired_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'absent'
            args = augment.parser().parse_args(['--output', str(root), '--until', '2000-01-01T00:00:00+09:00'])
            augment.run(args)
            self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
