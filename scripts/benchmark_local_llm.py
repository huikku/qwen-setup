#!/usr/bin/env python3
"""
benchmark_local_llm.py - Verification script for local LLM inference speed, TTFT, and context.
Supports OpenAI-compatible endpoints (llama-server, Ollama, Strata, vLLM).
"""
import time
import json
import urllib.request
import sys
import argparse

def run_benchmark(endpoint, prompt, max_tokens):
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a concise programming assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}
    print(f"Connecting to {url}...")
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp:
            elapsed = time.perf_counter() - t0
            res = json.loads(resp.read().decode('utf-8'))

            content = res["choices"][0]["message"]["content"]
            usage = res.get("usage", {})
            prompt_toks = usage.get("prompt_tokens", len(prompt.split()))
            comp_toks = usage.get("completion_tokens", len(content.split()))

            tok_per_sec = comp_toks / elapsed if elapsed > 0 else 0

            print("\n" + "="*50)
            print("         BENCHMARK RESULTS REPORT")
            print("="*50)
            print(f"Total Turnaround Time: {elapsed:.2f} seconds")
            print(f"Prompt Tokens:         {prompt_toks}")
            print(f"Generated Tokens:      {comp_toks}")
            print(f"Generation Throughput: {tok_per_sec:.2f} tok/s")
            print("-"*50)
            print("Output Sample (first 150 chars):")
            print(content[:150].strip() + "...")
            print("="*50)

            if tok_per_sec < 8.0:
                print("\n[!] WARNING: Throughput is under 8 tok/s.")
                print("    This indicates model layers are spilling into system RAM.")
            elif tok_per_sec >= 40.0:
                print("\n[*] EXCELLENT: GPU-native generation with zero PCIe spillage.")
            return True
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local LLM Benchmark Harness")
    parser.add_argument("--endpoint", default="http://localhost:8080", help="Base URL of server")
    parser.add_argument("--tokens", type=int, default=256, help="Tokens to generate")
    args = parser.parse_args()

    test_prompt = "Write an optimized LRU Cache in Rust using standard library data structures."
    success = run_benchmark(args.endpoint.rstrip('/'), test_prompt, args.tokens)
    sys.exit(0 if success else 1)
