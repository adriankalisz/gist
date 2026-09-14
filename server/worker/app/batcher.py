import asyncio
from app.config import settings
from app.summarizer import summarize


request_queue: asyncio.Queue = asyncio.Queue()


async def enqueue_request(article, num_beams: int | None = None):
    if num_beams is None:
        num_beams = settings.num_beams
    future = asyncio.get_running_loop().create_future()
    await request_queue.put((article, num_beams, future))
    return await future


async def batch_summarize():
    loop = asyncio.get_running_loop()
    while True:
        articles: list[str] = []
        futures: list[asyncio.Future] = []
        num_beams = settings.num_beams

        try:
            # Block until the first item arrives - this is what defines a batch
            # and avoids busy-looping while the queue is idle.
            article, num_beams, future = await request_queue.get()
            articles.append(article)
            futures.append(future)
            deadline = loop.time() + settings.batch_wait_ms / 1000

            # Keep collecting more items until the batch is full or batch_wait_ms
            # has elapsed since the first item - whichever comes first.
            while len(articles) < settings.summarization_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    article, _, future = await asyncio.wait_for(request_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                articles.append(article)
                futures.append(future)

            try:
                # Run the blocking model call off the event loop so new
                # requests can still be accepted/enqueued while it runs.
                summaries, inference_time = await asyncio.to_thread(summarize, articles, num_beams)
            except Exception as exc:
                for f in futures:
                    if not f.done():
                        f.set_exception(exc)
                continue

            for f, summary in zip(futures, summaries):
                if not f.done():
                    f.set_result((summary, inference_time))

        except asyncio.CancelledError:
            # Server is shutting down: fail whatever was already collected into
            # this batch, plus anything still waiting in the queue, so callers
            # get a clean error instead of hanging forever. Note: if cancellation
            # lands mid-inference, the model.generate() call keeps running to
            # completion in its background thread - its result is just discarded.
            shutdown_exc = RuntimeError("server shutting down")
            for f in futures:
                if not f.done():
                    f.set_exception(shutdown_exc)
            while not request_queue.empty():
                _, _, f = request_queue.get_nowait()
                if not f.done():
                    f.set_exception(shutdown_exc)
            raise
