"""NDJSON transport for incremental tutor responses."""
import json
import ai_cost_test as cost
from jiter import from_json
from fastapi.responses import StreamingResponse


def stream_chat(create, metadata=None, structured=False):
    def events():
        upstream = None
        try:
            upstream = create()
            content = ""
            displayed = ""
            finished = False
            for chunk in upstream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    if choice.finish_reason != "stop":
                        raise ValueError("Incomplete response")
                    finished = True
                content += choice.delta.content or ""
                text = content
                if structured:
                    clean = content.strip().removeprefix("```json").removeprefix("```").strip()
                    try:
                        partial = from_json(clean.encode(), partial_mode="trailing-strings")
                        text = "\n\n".join(partial.get(key, "") for key in ("feedback", "explanation", "hint") if isinstance(partial.get(key), str))
                    except ValueError:
                        text = ""
                if text and text != displayed:
                    displayed = text
                    yield json.dumps({"type": "text", "text": text}) + "\n"
            if not finished or not content.strip():
                raise ValueError("Incomplete response")
            if structured:
                data = json.loads(content.replace("```json", "").replace("```", "").strip())
                if data.get("evaluation") not in ("correct", "partial", "incorrect") or not isinstance(data.get("score"), (float, int)):
                    raise ValueError("Invalid evaluation")
            else:
                data = {"answer": content, **(metadata or {})}
            yield json.dumps({"type": "done", "data": data}) + "\n"
        except Exception:
            cost.outcome("failed")
            yield json.dumps({"type": "error", "message": "Response interrupted. Please try again."}) + "\n"
        finally:
            if upstream is not None:
                upstream.close()
    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
