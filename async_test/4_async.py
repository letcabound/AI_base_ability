# -*- coding: utf-8 -*-
from fastapi import FastAPI, BackgroundTasks
import asyncio
import time

app = FastAPI()

# --- 模拟耗时异步任务 ---
async def async_task(name: str, delay: int):
    print(f"{name} start at {time.time():.2f}")
    await asyncio.sleep(delay)  # 非阻塞挂起
    print(f"{name} done at {time.time():.2f}")

# --- 模拟耗时阻塞任务 ---
def blocking_task(name: str, delay: int):
    print(f"{name} start at {time.time():.2f}")
    time.sleep(delay)  # 阻塞线程
    print(f"{name} done at {time.time():.2f}")

# ----------------------
# 方案 1：直接异步路由 await
# ----------------------
@app.get("/async_wait")
async def async_wait_route():
    # 请求协程挂起，等待任务完成
    await async_task("AsyncTask1", 2)
    return {"message": "Async task finished"}

# ----------------------
# 方案 2：后台任务（非阻塞请求返回）
# ----------------------
@app.get("/background_task")
async def background_task_route(background_tasks: BackgroundTasks):
    # 注册后台异步任务，主协程立即返回
    background_tasks.add_task(async_task, "BGTask1", 3)
    return {"message": "Request returned, background task running"}

# ----------------------
# 方案 3：后台阻塞任务（线程池）
# ----------------------
@app.get("/background_blocking")
async def background_blocking_route(background_tasks: BackgroundTasks):
    # 阻塞任务必须放线程池 / to_thread，否则会卡住事件循环
    background_tasks.add_task(asyncio.to_thread, blocking_task, "BGBlocking1", 3)
    return {"message": "Request returned, blocking task running in thread"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)