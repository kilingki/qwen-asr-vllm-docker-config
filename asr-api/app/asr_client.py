from pathlib import Path
from typing import Any

import httpx


class QwenAsrClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self.client.aclose()

    async def list_models(self) -> dict[str, Any]:
        response = await self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

    async def transcribe(
        self,
        audio_path: Path,
        model: str,
        language: str | None,
        prompt: str | None,
        timestamp_granularities: list[str],
    ) -> dict[str, Any]:
        return await self._transcribe_request(
            audio_path=audio_path,
            model=model,
            language=language,
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
        )

    async def _transcribe_request(
        self,
        audio_path: Path,
        model: str,
        language: str | None,
        prompt: str | None,
        timestamp_granularities: list[str],
    ) -> dict[str, Any]:
        del timestamp_granularities

        form_data = {
            "model": model,
            "response_format": "json",
        }
        if language:
            form_data["language"] = language
        if prompt:
            form_data["prompt"] = prompt

        audio_bytes = audio_path.read_bytes()
        files = {
            "file": (audio_path.name, audio_bytes, "audio/wav"),
        }
        response = await self.client.post(
            f"{self.base_url}/audio/transcriptions",
            data=form_data,
            files=files,
        )

        if response.is_error:
            raise httpx.HTTPStatusError(
                (
                    f"ASR backend returned status={response.status_code}: "
                    f"{response.text}"
                ),
                request=response.request,
                response=response,
            )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            return {"text": str(payload)}

        return {"text": response.text}
