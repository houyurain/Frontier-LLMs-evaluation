import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Tuple

from openai import OpenAI


def setup_logging(verbose: bool = False):
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(
		level=level,
		format="[%(asctime)s] %(levelname)s %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)


def get_client() -> OpenAI:
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		logging.error("OPENAI_API_KEY is not set in environment.")
		sys.exit(1)
	return OpenAI(api_key=api_key)


def ensure_dirs():
	os.makedirs("logs/batch", exist_ok=True)
	os.makedirs("results/batch", exist_ok=True)


def _write_state(record: dict):
	ensure_dirs()
	path = os.path.join("logs", "batch", "batches.log")
	with open(path, "a", encoding="utf-8") as f:
		f.write(json.dumps(record, ensure_ascii=False) + "\n")


def upload_file(client: OpenAI, input_path: str) -> str:
	if not os.path.isfile(input_path):
		raise FileNotFoundError(f"Input file not found: {input_path}")

	# OpenAI Batch API expects JSONL with one request per line
	# We do a quick sanity check for the first line
	with open(input_path, "r", encoding="utf-8") as f:
		first_line = f.readline().strip()
		if first_line:
			try:
				obj = json.loads(first_line)
				# Minimal shape check
				if not all(k in obj for k in ("custom_id", "method", "url", "body")):
					logging.warning(
						"Input does not look like a batch JSONL (missing custom_id/method/url/body). Proceeding anyway."
					)
			except json.JSONDecodeError:
				logging.warning("Input file is not JSONL (first line not JSON). Proceeding anyway.")

	logging.info(f"Uploading file to OpenAI: {input_path}")
	with open(input_path, "rb") as f:
		uploaded = client.files.create(file=f, purpose="batch")
	logging.info(f"Uploaded file_id: {uploaded.id}")
	return uploaded.id


def submit_batch(
	input_path: str,
	model: Optional[str],
	completion_window: str = "24h",
	endpoint: str = "/v1/responses",
	metadata: Optional[dict] = None,
) -> str:
	client = get_client()
	file_id = upload_file(client, input_path)

	# Note: model is typically provided in each JSONL body for /v1/responses or /v1/chat/completions.
	# You can still track it in metadata for convenience.
	md = metadata or {}
	if model:
		md.setdefault("model", model)
	md.setdefault("submitted_from", os.uname().nodename)

	logging.info(
		f"Creating batch (endpoint={endpoint}, completion_window={completion_window})"
	)
	batch = client.batches.create(
		input_file_id=file_id,
		endpoint=endpoint,
		completion_window=completion_window,
		metadata=md,
	)

	record = {
		"ts": datetime.utcnow().isoformat() + "Z",
		"action": "submit",
		"batch_id": batch.id,
		"input_file_id": file_id,
		"endpoint": endpoint,
		"completion_window": completion_window,
		"metadata": md,
		"input_path": os.path.abspath(input_path),
	}
	_write_state(record)

	logging.info(f"Submitted batch_id: {batch.id}")
	return batch.id


def get_status(batch_id: str) -> dict:
	client = get_client()
	batch = client.batches.retrieve(batch_id)
	# Convert to plain dict for printing
	data = json.loads(batch.model_dump_json()) if hasattr(batch, "model_dump_json") else batch
	return data


def list_batches(limit: int = 20) -> dict:
	client = get_client()
	batches = client.batches.list(limit=limit)
	data = json.loads(batches.model_dump_json()) if hasattr(batches, "model_dump_json") else batches
	return data


def cancel_batch(batch_id: str) -> dict:
	client = get_client()
	result = client.batches.cancel(batch_id)
	data = json.loads(result.model_dump_json()) if hasattr(result, "model_dump_json") else result
	record = {
		"ts": datetime.utcnow().isoformat() + "Z",
		"action": "cancel",
		"batch_id": batch_id,
		"result": data,
	}
	_write_state(record)
	return data


def download_output(batch_id: str, out_path: Optional[str] = None) -> str:
	client = get_client()
	batch = client.batches.retrieve(batch_id)

	if getattr(batch, "status", None) != "completed":
		raise RuntimeError(f"Batch {batch_id} not completed. Current status: {getattr(batch, 'status', 'unknown')}")

	output_file_id = getattr(batch, "output_file_id", None)
	if not output_file_id:
		raise RuntimeError(f"No output_file_id found for batch {batch_id}")

	ensure_dirs()
	if not out_path:
		ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
		out_path = os.path.join("results", "batch", f"{batch_id}_{ts}.jsonl")

	logging.info(f"Downloading output file {output_file_id} to {out_path}")

	# Prefer streaming if available for large files
	try:
		with client.files.with_streaming_response.content(output_file_id) as stream:
			stream.stream_to_file(out_path)
	except Exception:
		# Fallback: non-streaming
		content = client.files.content(output_file_id)
		# Try common attributes across SDK versions
		data = getattr(content, "text", None)
		if data is None:
			data = getattr(content, "content", None)
		if data is None and hasattr(content, "read"):
			data = content.read()
		if isinstance(data, (bytes, bytearray)):
			with open(out_path, "wb") as f:
				f.write(data)
		else:
			text = data if isinstance(data, str) else str(content)
			with open(out_path, "w", encoding="utf-8") as f:
				f.write(text)

	record = {
		"ts": datetime.utcnow().isoformat() + "Z",
		"action": "download",
		"batch_id": batch_id,
		"output_file_id": output_file_id,
		"out_path": os.path.abspath(out_path),
	}
	_write_state(record)
	return out_path


def download_error(batch_id: str, err_path: Optional[str] = None) -> str:
	"""Download the error file for a batch if present. Works regardless of batch status.

	Returns the path written to. Raises if no error_file_id is present.
	"""
	client = get_client()
	batch = client.batches.retrieve(batch_id)

	error_file_id = getattr(batch, "error_file_id", None)
	if not error_file_id:
		raise RuntimeError(f"No error_file_id found for batch {batch_id}")

	ensure_dirs()
	if not err_path:
		ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
		err_path = os.path.join("results", "batch", f"{batch_id}_{ts}.errors.jsonl")

	logging.info(f"Downloading error file {error_file_id} to {err_path}")

	# Prefer streaming if available for large files
	try:
		with client.files.with_streaming_response.content(error_file_id) as stream:
			stream.stream_to_file(err_path)
	except Exception:
		# Fallback: non-streaming
		content = client.files.content(error_file_id)
		data = getattr(content, "text", None)
		if data is None:
			data = getattr(content, "content", None)
		if data is None and hasattr(content, "read"):
			data = content.read()
		if isinstance(data, (bytes, bytearray)):
			with open(err_path, "wb") as f:
				f.write(data)
		else:
			text = data if isinstance(data, str) else str(content)
			with open(err_path, "w", encoding="utf-8") as f:
				f.write(text)

	record = {
		"ts": datetime.utcnow().isoformat() + "Z",
		"action": "download_error",
		"batch_id": batch_id,
		"error_file_id": error_file_id,
		"err_path": os.path.abspath(err_path),
	}
	_write_state(record)
	return err_path


def wait(batch_id: str, poll_secs: int = 30, timeout_secs: int = 0):
	"""Poll batch status until completion or failure. Set timeout_secs=0 to wait indefinitely."""
	start = time.time()
	while True:
		data = get_status(batch_id)
		status = data.get("status")
		logging.info(f"Batch {batch_id} status: {status}")
		if status in {"completed", "failed", "cancelled", "expired"}:
			return data
		if timeout_secs and (time.time() - start) > timeout_secs:
			raise TimeoutError(f"Timed out after {timeout_secs}s waiting for batch {batch_id}")
		time.sleep(poll_secs)


def build_arg_parser():
	p = argparse.ArgumentParser(
		description="Submit and manage OpenAI Batch API jobs using local JSONL files.")
	p.add_argument("command", choices=["submit", "status", "list", "cancel", "download", "wait", "retrieve_all", "cancel_all"], help="Action to perform")
	p.add_argument("--input", dest="input_path", help="Path to local JSONL for submit")
	p.add_argument("--model", dest="model", default=None, help="Optional model tag to record in metadata (each JSONL line should still include a model in body)")
	p.add_argument("--endpoint", default="/v1/responses", help="API endpoint for the batch: /v1/responses or /v1/chat/completions")
	p.add_argument("--window", dest="completion_window", default="24h", help="Completion window: 24h or 48h (if available)")
	p.add_argument("--batch_id", dest="batch_id", help="Existing batch id for status/cancel/download/wait")
	p.add_argument("--out", dest="out_path", help="Output path for download")
	p.add_argument("--limit", type=int, default=20, help="Limit for list")
	p.add_argument("--poll", dest="poll_secs", type=int, default=30, help="Polling interval for wait")
	p.add_argument("--timeout", dest="timeout_secs", type=int, default=0, help="Timeout in seconds for wait (0 = no timeout)")
	p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
	# retrieve_all options
	p.add_argument("--log", dest="log_file", default=os.path.join("logs", "batch", "batches.log"), help="Path to log file (used by retrieve_all and cancel_all)")
	p.add_argument("--force", dest="force", action="store_true", help="Overwrite existing outputs when running retrieve_all")
	# cancel_all options
	p.add_argument("--dry_run", dest="dry_run", action="store_true", help="List batch IDs found in --log without cancelling (for cancel_all)")
	return p


def main(argv=None):
	args = build_arg_parser().parse_args(argv)
	setup_logging(args.verbose)
	ensure_dirs()

	if args.command == "submit":
		if not args.input_path:
			logging.error("--input is required for submit")
			sys.exit(2)
		batch_id = submit_batch(
			input_path=args.input_path,
			model=args.model,
			completion_window=args.completion_window,
			endpoint=args.endpoint,
		)
		print(batch_id)
		return

	if args.command == "status":
		if not args.batch_id:
			logging.error("--batch-id is required for status")
			sys.exit(2)
		data = get_status(args.batch_id)
		print(json.dumps(data, indent=2))
		return

	if args.command == "list":
		data = list_batches(limit=args.limit)
		print(json.dumps(data, indent=2))
		return

	if args.command == "cancel":
		if not args.batch_id:
			logging.error("--batch-id is required for cancel")
			sys.exit(2)
		data = cancel_batch(args.batch_id)
		print(json.dumps(data, indent=2))
		return

	if args.command == "download":
		if not args.batch_id:
			logging.error("--batch-id is required for download")
			sys.exit(2)
		out_path = download_output(args.batch_id, args.out_path)
		print(out_path)
		# also download error file if present
		try:
			err_path = download_error(args.batch_id)
			logging.info(f"Also downloaded error file to: {err_path}")
		except Exception as e:
			logging.info(f"No error file downloaded: {e}")
		return

	if args.command == "wait":
		if not args.batch_id:
			logging.error("--batch-id is required for wait")
			sys.exit(2)
		data = wait(args.batch_id, poll_secs=args.poll_secs, timeout_secs=args.timeout_secs)
		print(json.dumps(data, indent=2))
		return

	if args.command == "retrieve_all":
		def _read_submit_records(log_path: str) -> Dict[str, Tuple[str, dict]]:
			"""
			Return mapping batch_id -> (input_path, metadata) from submit records in batches.log.
			If multiple records exist for same batch_id, keep the last occurrence.
			"""
			records: Dict[str, Tuple[str, dict]] = {}
			if not os.path.isfile(log_path):
				logging.error(f"Log file not found: {log_path}")
				return records
			with open(log_path, "r", encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if not line:
						continue
					try:
						obj = json.loads(line)
					except json.JSONDecodeError:
						continue
					if obj.get("action") == "submit":
						bid = obj.get("batch_id")
						inp = obj.get("input_path")
						md = obj.get("metadata", {})
						if bid and inp:
							records[bid] = (inp, md)
			return records

		def _derive_out_path(input_path: str) -> str:
			# Store alongside the input JSONL, same name but .json
			dirname, fname = os.path.dirname(input_path), os.path.basename(input_path)
			if fname.endswith(".jsonl"):
				fname = fname[:-6] + ".json"
			else:
				fname = fname + ".json"
			os.makedirs(dirname, exist_ok=True)
			return os.path.join(dirname, fname)

		def _derive_err_path(input_path: str) -> str:
			# Store alongside the input JSONL, same name but .errors.json
			dirname, fname = os.path.dirname(input_path), os.path.basename(input_path)
			if fname.endswith(".jsonl"):
				fname = fname[:-6] + ".errors.json"
			else:
				fname = fname + ".errors.json"
			os.makedirs(dirname, exist_ok=True)
			return os.path.join(dirname, fname)

		submit_map = _read_submit_records(args.log_file)
		if not submit_map:
			logging.info("No submit records found to retrieve.")
			print("[]")
			return

		client = get_client()
		results = []
		for bid, (inp, md) in submit_map.items():
			try:
				batch = client.batches.retrieve(bid)
				status = getattr(batch, "status", None)
				out_path = _derive_out_path(inp)
				err_path = _derive_err_path(inp)
				entry = {
					"batch_id": bid,
					"status": status,
					"input_path": inp,
					"out_path": out_path,
					"error_path": err_path,
				}
				if status == "completed":
					if (not args.force) and os.path.exists(out_path):
						logging.info(f"Skip existing: {out_path}")
						results.append(entry)
						continue
					# Use existing download_output to save file to desired path
					try:
						# Reuse function to try both streaming and fallback
						_ = download_output(bid, out_path)
						entry["saved"] = True
						logging.info(f"Saved output for {bid} -> {out_path}")
					except Exception as e:
						entry["saved"] = False
						entry["error"] = str(e)
				else:
					entry["saved"] = False
					entry["error"] = f"Not completed (status={status})"

				# Independently attempt to download error file if present
				error_file_id = getattr(batch, "error_file_id", None)
				if error_file_id:
					if (not args.force) and os.path.exists(err_path):
						logging.info(f"Skip existing error file: {err_path}")
						entry["saved_error"] = True
					else:
						try:
							_ = download_error(bid, err_path)
							entry["saved_error"] = True
							entry["error_file_id"] = error_file_id
							logging.info(f"Saved error file for {bid} -> {err_path}")
						except Exception as e:
							entry["saved_error"] = False
							entry["error_error"] = str(e)
				else:
					entry["saved_error"] = False
			except Exception as e:
				results.append({"batch_id": bid, "error": str(e), "input_path": inp})
				continue
			results.append(entry)

		print(json.dumps(results, indent=2))
		return

	if args.command == "cancel_all":
		import re

		BATCH_ID_RE = re.compile(r"batch_[0-9a-f]+", re.IGNORECASE)

		def _collect_from_obj(obj, sink: Dict[str, dict]):
			# Direct batch_id
			bid = obj.get("batch_id") if isinstance(obj, dict) else None
			if isinstance(bid, str):
				sink.setdefault(bid, {})
			# Object is a batch (from API), usually has id and object=="batch"
			if isinstance(obj, dict) and obj.get("object") == "batch" and isinstance(obj.get("id"), str):
				sink.setdefault(obj["id"], {})
			# Submit record support
			if isinstance(obj, dict) and obj.get("action") == "submit":
				bid2 = obj.get("batch_id")
				if isinstance(bid2, str):
					sink.setdefault(bid2, {"input_path": obj.get("input_path"), "metadata": obj.get("metadata", {})})
			# List results
			if isinstance(obj, dict) and isinstance(obj.get("data"), list):
				for it in obj["data"]:
					if isinstance(it, dict) and isinstance(it.get("id"), str):
						sink.setdefault(it["id"], {})

		def _read_batch_ids_generic(log_path: str) -> Dict[str, dict]:
			ids: Dict[str, dict] = {}
			if not os.path.isfile(log_path):
				logging.error(f"Log file not found: {log_path}")
				return ids
			with open(log_path, "r", encoding="utf-8") as f:
				for line in f:
					text = line.strip()
					if not text:
						continue
					# Try JSON first
					parsed = None
					try:
						parsed = json.loads(text)
					except json.JSONDecodeError:
						parsed = None
					if isinstance(parsed, dict):
						_collect_from_obj(parsed, ids)
					elif isinstance(parsed, list):
						for item in parsed:
							if isinstance(item, dict):
								_collect_from_obj(item, ids)
					# Regex fallback for non-JSON lines
					for m in BATCH_ID_RE.findall(text):
						ids.setdefault(m, {})
			return ids

		ids = _read_batch_ids_generic(args.log_file)
		if not ids:
			logging.info("No batch ids found to cancel.")
			print("[]")
			return

		if args.dry_run:
			out = sorted(ids.keys())
			print(json.dumps({"count": len(out), "batch_ids": out}, indent=2))
			return

		results = []
		for bid, info in ids.items():
			entry = {"batch_id": bid, **info}
			try:
				resp = cancel_batch(bid)
				entry.update({
					"cancelled": True,
					"result": resp,
				})
			except Exception as e:
				entry.update({
					"cancelled": False,
					"error": str(e),
				})
			results.append(entry)

		print(json.dumps(results, indent=2))
		return


if __name__ == "__main__":
	main()

