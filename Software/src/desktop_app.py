"""VIDS Desktop launcher: embedded Flask backend + native window.

Frozen (PyInstaller): data lives in sys._MEIPASS (templates, static,
training modules, model, ffmpeg). Uploads/spectrograms are transient
and stay in the bundle temp dir (writable, cleaned on exit).
Dev: runs against the source tree via VIDS_ROOT default in app.py.
"""
import os
import sys
import threading
import time
import urllib.request

PORT = 5000
APP_TITLE = 'VIDS - Voice Impersonation Detection System'


if not getattr(sys, 'frozen', False):
    # dev: make webapp + training importable (frozen uses _prepare_frozen instead)
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _sub in ('webapp', 'training'):
        _p = os.path.join(_ROOT, _sub)
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)


def _bundle_dir():
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))


def _content_dir():
    """Directory holding bundled data (PyInstaller one-dir puts datas under
    _internal/, while sys._MEIPASS points at the app dir)."""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        inner = os.path.join(base, '_internal')
        if os.path.isdir(inner):
            return inner
        return base
    return None


def _log(msg):
    try:
        with open(os.path.join(os.environ.get('TEMP', '.'), 'VIDS_debug.log'),
                  'a', encoding='utf-8') as f:
            f.write(msg + '\n')
    except Exception:
        pass
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _prepare_frozen():
    if not getattr(sys, 'frozen', False):
        return
    # windowed exe has no console: route ALL stdout/stderr to a log file so
    # no print anywhere (app, libs, traceback) can crash the process
    try:
        _logfile = open(os.path.join(os.environ.get('TEMP', '.'), 'VIDS_stdout.log'),
                        'a', encoding='utf-8', buffering=1)
        sys.stdout = _logfile
        sys.stderr = _logfile
    except Exception:
        pass
    base = _content_dir() or sys._MEIPASS
    _log(f'[VIDS] bundle content dir: {base}')
    # bundled python modules
    for sub in ('training', 'webapp'):
        p = os.path.join(base, sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    # app.py resolves data from VIDS_ROOT
    os.environ.setdefault('VIDS_ROOT', base)
    # bundled ffmpeg for webm decode + duration probe
    if base not in os.environ.get('PATH', ''):
        os.environ['PATH'] = base + os.pathsep + os.environ.get('PATH', '')
    # silence GPU-fork warnings on some machines
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def _wait_for_backend(url, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    _prepare_frozen()
    _log('[VIDS] starting desktop app')
    import app as vids_app  # noqa: E402  (paths fixed above)

    _log('[VIDS] loading Professor K model...')
    vids_app.load_model()
    _log('[VIDS] model ready, starting backend')

    if getattr(sys, 'frozen', False):
        # windowed exe has no stderr: silence werkzeug request logs or every
        # request dies writing to the closed stream
        import logging as _logging
        _logging.getLogger('werkzeug').setLevel(_logging.ERROR)

    def run_server():
        # threaded: SSE stream + page loads concurrently
        vids_app.app.run(host='127.0.0.1', port=PORT, debug=False,
                         threaded=True, use_reloader=False)

    threading.Thread(target=run_server, daemon=True).start()
    if not _wait_for_backend(f'http://127.0.0.1:{PORT}/health'):
        _log('[VIDS] ERROR: backend failed to start.')
        return 1
    _log('[VIDS] backend up, opening window')

    import webview  # noqa: E402
    webview.create_window(APP_TITLE, f'http://127.0.0.1:{PORT}',
                          width=1220, height=880, min_size=(940, 640))
    webview.start()
    return 0


if __name__ == '__main__':
    sys.exit(main())
