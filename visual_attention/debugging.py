# from tensorflow.python.training.training import SessionManager
from tensorflow.python import debug as tf_debug
from tensorflow.python.training.session_manager import SessionManager


class DebuggableSessionManager(SessionManager):
    def __init__(self, debug, *args, **kwargs):
        self.debug = debug
        super(SessionManager).__init__(*args, **kwargs)

    def prepare_session(self, *args, **kwargs):
        sess = super(SessionManager).prepare_session(*args, **kwargs)
        if self.debug:
            print("Debugging session created!!!!!!!!!")
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        return sess


def enable_debugging_monkey_patch():
    old_prepare_session = SessionManager.prepare_session

    def new_prepare_session(self, *args, **kwargs):
        sess = old_prepare_session(self, *args, **kwargs)
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        return sess

    SessionManager.prepare_session = new_prepare_session
