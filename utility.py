# Refer from: https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/serialize.py


import msgpack
import msgpack_numpy

msgpack_numpy.patch()

PREDICT_CMD = "Predict"
TRAIN_CMD   = "Train"

LRU_READY = "\x01"            # Server side worker READY CHAR

def dumps(obj):
    """
    Serialize an object.

    Returns:
        str
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads(buf):
    """
    Args:
        buf (str): serialized object.
    """
    return msgpack.loads(buf)