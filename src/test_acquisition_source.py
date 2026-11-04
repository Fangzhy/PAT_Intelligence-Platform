"""Standalone, loopback-only historical or paced fresh test acquisition source.

Run: python -m src.test_acquisition_source --port 8765
This server exposes only fixed JSON records and a health endpoint, not files.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import parse_qs, urlsplit

from src.acquisition import AcquisitionError, InMemoryTestAdapter, test_source_records
from src.data_generation import SimulationConfig, generate_dataset
from src.data_integration import sources_from_dataset
from src.test_stream_source import PacedTestAdapter


def make_test_server(adapter, port=0):
    """Create a server; callers own serve_forever/shutdown/server_close."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_json(self, value, status=200):
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/health":
                self.send_json({"status": "ok", "source_id": adapter.source_id, "records": len(adapter.records), "mode": "paced fresh synthetic test source" if isinstance(adapter, PacedTestAdapter) else "historical synthetic test source"})
                return
            if url.path != "/v1/records":
                self.send_json({"error": "Unknown test-source endpoint"}, 404)
                return
            try:
                query = parse_qs(url.query, strict_parsing=True)
                if set(query) != {"after", "limit"} or any(len(values) != 1 for values in query.values()):
                    raise AcquisitionError("Supply exactly one after cursor and limit.")
                page = adapter.poll(int(query["after"][0]), int(query["limit"][0]))
                self.send_json(page.as_dict())
            except (ValueError, AcquisitionError) as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self):
            self.send_json({"error": "This test source is read-only"}, 405)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="Serve historical or paced fresh synthetic test records on loopback.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=24, help="Synthetic batch duration in hours")
    parser.add_argument("--channels", type=int, default=71)
    parser.add_argument("--faults", action="store_true", help="Duplicate delivery, invalid time/schema, and missing pH examples")
    parser.add_argument("--stream", action="store_true", help="Release fresh-timestamp records over real time")
    parser.add_argument("--seconds-per-hour", type=float, default=1.0, help="Real seconds per original simulated hour in --stream mode")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        config = SimulationConfig(n_batches=1, duration_h=args.duration, n_wavenumbers=args.channels, seed=args.seed)
        records = test_source_records(sources_from_dataset(generate_dataset(config)), faults=args.faults)
        adapter = PacedTestAdapter(records, seconds_per_hour=args.seconds_per_hour) if args.stream else InMemoryTestAdapter(records)
        server = make_test_server(adapter, args.port)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Test source: http://127.0.0.1:{args.port} · {len(adapter.records)} records · {adapter.source_id}", flush=True)
    print("Fresh UTC timestamps, accelerated synthetic process." if args.stream else "Historical simulation timestamps; receipt time is current UTC.", flush=True)
    print("Ctrl+C stops the source.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
