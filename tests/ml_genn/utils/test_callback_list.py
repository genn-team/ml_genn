from ml_genn.callbacks import Callback
from ml_genn.utils.callback_list import CallbackList

from pytest import raises, warns

def test_missing_key():
    # Create fake callback which returns None as it's data key
    class NoKeyCallback(Callback):
        def get_data(self):
            return None, "hello"

    # Place one in callback list
    cb_list = CallbackList([NoKeyCallback()])

    # Check that warning is emitted when getting data
    with warns():
        data = cb_list.get_data()

def test_duplicate_key():
    # Create fake callback which returns fixed key
    class FixedKeyCallback(Callback):
        def get_data(self):
            return "key", "hello"

    # Place one in callback list
    cb_list = CallbackList([FixedKeyCallback(), FixedKeyCallback()])

    # Check that exception is raised when getting data
    with raises(KeyError):
        data = cb_list.get_data()
