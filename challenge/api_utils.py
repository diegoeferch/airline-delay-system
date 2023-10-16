from fastapi import Request
from starlette.responses import JSONResponse


async def base_exception_handler(request: Request, exc: BaseException):
    return JSONResponse(status_code=400, content={})

