import asyncio

class AsyncStdioWrapper:
    def __init__(self, fileobj, mode: str):
        self._file = fileobj
        self._mode = mode
        self._loop = None
        self._closed = False

    async def __aenter__(self):
        self._loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._closed = True
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        data = await self.readline()
        if not data:
            raise StopAsyncIteration
        return data

    async def read(self, n: int = -1):
        if self._closed:
            return b""
        loop = self._loop or asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._file.read, n)

    async def readline(self):
        if self._closed:
            return b""
        loop = self._loop or asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._file.readline)

    async def readexactly(self, n: int):
        if self._closed:
            return b""
        loop = self._loop or asyncio.get_running_loop()

        def _readexactly(cnt):
            data = b''
            while len(data) < cnt:
                chunk = self._file.read(cnt - len(data))
                if not chunk:
                    break
                data += chunk
            return data

        return await loop.run_in_executor(None, _readexactly, n)

    async def write(self, data: bytes):
        if self._closed:
            return 0
        loop = self._loop or asyncio.get_running_loop()

        def _write(d):
            written = self._file.write(d)
            try:
                self._file.flush()
            except Exception:
                pass
            return written

        return await loop.run_in_executor(None, _write, data)

    async def drain(self):
        await asyncio.sleep(0)
