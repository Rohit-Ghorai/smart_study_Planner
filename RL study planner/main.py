import os
import threading
import webbrowser

from study_planner.web import app


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, webbrowser.open_new_tab, args=(f"http://127.0.0.1:{port}/",)).start()
    app.run(host=host, port=port, debug=True, use_reloader=True)
