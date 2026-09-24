import threading

# Shared runtime lock for one-time heavyweight model initialization.
MODEL_INIT_LOCK = threading.RLock()
