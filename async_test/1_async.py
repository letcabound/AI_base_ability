# -*- coding: utf-8 -*-
import asyncio
import time

async def task_func(name, delay):
    print(f"{name} start at {time.time():.2f}")
    await asyncio.sleep(delay)
    print(f"{name} end at {time.time():.2f}")
    return name

# ---- 场景 1：顺序 await ----
async def sequential():
    print("\n--- Sequential await ---")
    start = time.time()
    await task_func("A", 1)
    await task_func("B", 1)
    end = time.time()
    print(f"Sequential done in {end - start:.2f} s") # 2

# ---- 场景 2：create_task 不 await (fire-and-forget) ----
async def fire_and_forget():
    print("\n--- create_task (no await) ---")
    start = time.time()
    asyncio.create_task(task_func("C", 1))
    asyncio.create_task(task_func("D", 1))
    end = time.time()
    print(f"Fire-and-forget main done in {end - start:.2f} s") # 0
    await asyncio.sleep(1.5)  # 给后台任务时间执行

# ---- 场景 3：create_task + await ----
async def tasks_with_await():
    print("\n--- create_task + await ---")
    start = time.time()
    t1 = asyncio.create_task(task_func("E", 1))
    t2 = asyncio.create_task(task_func("F", 1))
    await t1
    await t2
    end = time.time()
    print(f"Tasks with await done in {end - start:.2f} s") # 1

# ---- 场景 4：gather 并发等待 ----
async def gather_example():
    print("\n--- gather ---")
    start = time.time()
    results = await asyncio.gather(
        task_func("G", 1),
        task_func("H", 1)
    )
    end = time.time()
    print(f"Gather done in {end - start:.2f} s")
    print("Gather results:", results)

async def test():
    a = await task_func("I", 1) # 直接拿到协程返回结果
    print(f"结果返回: {a}")

    task = asyncio.create_task(task_func("J", 1))
    b = await task
    print(f"结果返回: {b}")

async def main():
    # await sequential()
    # await fire_and_forget()
    # await tasks_with_await()
    # await gather_example()
    await test()

# await vs create_task vs gather
asyncio.run(main())
