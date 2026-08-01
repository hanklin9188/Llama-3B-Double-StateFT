import contextlib
import io


@contextlib.contextmanager
def _silence_stdout_stderr():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield
