# -*- coding: utf-8 -*-
import asyncio
import time

# 异步任务：正确挂起
async def async_task(name, delay):
    print(f"{name} start at {time.time():.2f}")
    await asyncio.sleep(delay)  # 非阻塞挂起
    print(f"{name} end at {time.time():.2f}")

# 同步阻塞任务：会卡住事件循环
async def blocking_task(name, delay):
    print(f"{name} start at {time.time():.2f}")
    time.sleep(delay)  # ❌ 阻塞整个事件循环
    print(f"{name} end at {time.time():.2f}")

async def main():
    print("\n--- Async tasks (await asyncio.sleep) ---")
    start = time.time()
    await asyncio.gather(
        async_task("A", 1),
        async_task("B", 1),
    )
    end = time.time()
    print(f"Async tasks done in {end - start:.2f}s\n")

    print("--- Blocking tasks (time.sleep inside async) ---")
    start = time.time()
    await asyncio.gather(
        blocking_task("C", 1),
        blocking_task("D", 1),
    )
    end = time.time()
    print(f"Blocking tasks done in {end - start:.2f}s")

# 协程挂起 vs 阻塞代码
asyncio.run(main())
