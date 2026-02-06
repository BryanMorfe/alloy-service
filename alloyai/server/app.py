from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI

from .routes.audio import router as audio_router
from .routes.chat import router as chat_router
from .routes.image import router as image_router
from ..runtime import get_runtime


def create_app() -> FastAPI:
    app = FastAPI(title="Alloy")
    app.include_router(image_router)
    app.include_router(chat_router)
    app.include_router(audio_router)

    @app.on_event("startup")
    async def _startup() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=16))
        get_runtime()

    return app


app = create_app()
