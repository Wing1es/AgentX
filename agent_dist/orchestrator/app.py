import logging
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .schemas import QueryRequest, QueryResponse
from .dependencies import router, planner, executor
from .config import DEBUG_MODE
from .memory import Memory
import uuid

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("orchestrator")

memory = Memory()

app = FastAPI(
    title="Agent Orchestrator",
    description="Central brain for agent routing, planning, and execution",
    version="1.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    logger.info(f"[{trace_id}] Request: {request.method} {request.url}")
    
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"[{trace_id}] Response status: {response.status_code} took {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[{trace_id}] Request failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal Server Error",
                "trace_id": trace_id,
                "error": str(e) if DEBUG_MODE else "An unexpected error occurred."
            }
        )


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request):
    """
    Main entrypoint.
    Users ONLY call this.
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    logger.info(f"[{trace_id}] Processing query: {payload.query}")
    
    session_id = payload.session_id or str(uuid.uuid4())
    
    # 1. LOAD HISTORY
    history = memory.get_history(session_id)
    
    # 2. SAVE USER QUERY
    memory.add_message(session_id, "user", payload.query)

    total_start_time = time.time()

    # ROUTE
    decision = router.route(payload.query, history)

    # PLAN
    plan = planner.plan(decision)

    # EXECUTE
    result = await executor.execute(plan, payload.query, history)

    total_duration = time.time() - total_start_time
    result["total_duration_seconds"] = round(total_duration, 4)

    # 3. SAVE ASSISTANT RESPONSE
    final_answer = result.get("final_answer", "")
    if isinstance(final_answer, str):
        memory.add_message(session_id, "assistant", final_answer)

    # 4. SAVE TRACE
    trace_data = result.get("trace", [])
    if trace_data:
        memory.save_trace(session_id, trace_data)

    # RESPOND
    response = QueryResponse(
        answer=result,
        session_id=session_id,
        plan=plan.model_dump() if payload.debug or DEBUG_MODE else None
    )
    return response


def run():
    import uvicorn
    from .config import ORCHESTRATOR_PORT
    uvicorn.run(app, host="0.0.0.0", port=ORCHESTRATOR_PORT)

if __name__ == "__main__":
    run()
