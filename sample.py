def enqueue_submission(team_id: int, model_path: str) -> bool:
    """
    Try to enqueue without blocking. Returns True if queued, False if the queue is full.
    """
    try:
        submission_queue.put_nowait((team_id, model_path))
        print(f"[enqueue] queued team={team_id} {model_path} (size={submission_queue.qsize()})")
        return True
    except queue.Full:
        print(f"[enqueue] queue full (limit={submission_queue.maxsize}). "
              f"Rejecting team={team_id} {model_path}")
        return False