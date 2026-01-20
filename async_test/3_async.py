# -*- coding: utf-8 -*-
import asyncio
import time


# 正常协程
async def normal_task(name, delay):
    print(f"{name} start at {time.time():.2f}")
    await asyncio.sleep(delay)
    print(f"{name} done at {time.time():.2f}")
    return f"{name} result"


# 会抛异常的协程
async def error_task(name):
    print(f"{name} start")
    await asyncio.sleep(0.5)
    raise ValueError(f"{name} raised an error")


# 可取消的协程
async def cancellable_task(name):
    try:
        print(f"{name} start")
        await asyncio.sleep(2)
        print(f"{name} finished")
    except asyncio.CancelledError:
        print(f"{name} was cancelled")
        raise  # 如果需要让上层知道被取消


async def main():
    print("\n--- Task Lifecycle & Result ---")

    # 创建 Task，但暂不 await
    t1 = asyncio.create_task(normal_task("T1", 1))
    t2 = asyncio.create_task(normal_task("T2", 1))
    print("Tasks created, not awaited yet")

    # await t1，t2 还在 pending
    result1 = await t1
    print("Result1:", result1)

    # await t2
    result2 = await t2
    print("Result2:", result2)

    print("\n--- Task Exception Handling ---")
    t_err = asyncio.create_task(error_task("T_ERR"))
    try:
        await t_err
    except ValueError as e:
        print("Caught exception from task:", e)

    print("\n--- Task Cancellation ---")
    t_cancel = asyncio.create_task(cancellable_task("T_CANCEL"))
    await asyncio.sleep(0.5)
    t_cancel.cancel()  # 取消任务
    try:
        await t_cancel
    except asyncio.CancelledError:
        print("Cancellation confirmed in main")

# Task 生命周期 & 异常管理
asyncio.run(main())
