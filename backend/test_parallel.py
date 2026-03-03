import ollama
from concurrent.futures import ThreadPoolExecutor
import time

def test_call(n):
    print(f"Start {n}")
    start = time.time()
    res = ollama.chat(model="qwen2.5vl:3b", messages=[{"role": "user", "content": "say hi"}], keep_alive="1m")
    print(f"End {n} in {time.time()-start:.1f}s: {res['message']['content'][:20]}")

print("Testing parallel Ollama...")
with ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(test_call, range(4))
print("Done")
