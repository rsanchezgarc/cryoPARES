import numba
import numpy as np
from numba import njit, config as numba_conf
from numba.typed import Dict, List
from numba.core import types

# numba_conf.THREADING_LAYER = 'threadsafe'
# print(numba.config.NUMBA_NUM_THREADS)

@njit(cache=True, nogil=True)
def typed_dict_add_keyList_valList(typed_dict, key_list, values_list, dtype=types.int64):
    """
    Adds values to the lists associated with each key in the typed_dict.
    If a key does not exist, it initializes it with a new list.

    :param typed_dict: numba.typed.Dict, the dictionary to modify
    :param key_list: List of integers, the keys to modify
    :param values_list: List of integers, the values to add to each key's list
    """
    for k, v in zip(key_list, values_list):
        if k not in typed_dict:
            typed_dict[k] = List.empty_list(dtype)
        typed_dict[k].append(v)

@njit(cache=True, nogil=True)
def typed_dict_add_key_val_paired_lists(typed_dict, key_list, values_list, dtype=types.int64):
    """
    Adds values to the lists associated with each key in the typed_dict.
    If a key does not exist, it initializes it with a new list.

    :param typed_dict: numba.typed.Dict, the dictionary to modify
    :param key_list: List of integers, the keys to modify
    :param values_list: List of integers, the values to add to each key's list
    """
    for k, v in zip(key_list, values_list):
        typed_dict[k] = v

@njit(cache=True, nogil=True)
def typed_dict_read_values_from_keysList(typed_dict, key_list, dtype=types.int64):
    """
    Reads and returns the values associated with a list of keys from the typed_dict.

    :param typed_dict: numba.typed.Dict
    :param key_list: List of integers, the keys to read values from
    :return: A list of lists, each containing the values of the corresponding key
    """
    result = List.empty_list(dtype)
    for k in key_list:
        result.extend(typed_dict[k])
    return result

@njit(cache=True, nogil=True)
def typed_dict_read_keys(typed_dict, dtype=types.int64): #TODO: add to warm-up
    """
    Reads and returns the values associated with a list of keys from the typed_dict.

    :param typed_dict: numba.typed.Dict
    :param key_list: List of integers, the keys to read values from
    :return: A list of lists, each containing the values of the corresponding key
    """
    n = len(typed_dict.keys())
    result = np.empty(n, dtype=dtype)
    for i, k in enumerate(typed_dict.keys()):
        result[i] = k
    return result
@njit(cache=True, nogil=True)
def typed_dict_flatten_keys_vals_asarray(typed_dict, n_items=-1, sort=True):
    """
    Reads and returns the values associated with a list of keys from the typed_dict. The values are sorted by key

    :param typed_dict: numba.typed.Dict
    :param key_list: List of integers, the keys to read values from
    :return: A list of lists, each containing the values of the corresponding key
    """
    if n_items < 0:
        n_items = sum([len(x) for x in typed_dict.values()])
    keys = np.empty(n_items, dtype=np.int64)
    vals = np.empty(n_items, dtype=np.int64)

    keys_array = np.array(list(typed_dict.keys()), dtype=np.int64)
    if sort:
        keys_array = np.sort(keys_array)

    i = 0
    for k in keys_array:
        for v in typed_dict[k]:
            keys[i] = k
            vals[i] = v
            i += 1
    return keys, vals

@njit(cache=True, nogil=True, parallel=True)
def parallel_typed_dict_flatten_keys_vals_asarray(typed_dict, n_items=-1, sort=True, chunk_size=int(1e5)):
    """
    Flatten a Numba-typed dictionary into two numpy arrays, representing keys and values, respectively.
    This function processes the dictionary in chunks and is designed to be executed in parallel,
    improving performance on large datasets.

    Parameters:
    - typed_dict : numba.typed.Dict
      A Numba-typed dictionary where keys are integers and values are lists of integers.
    - n_items : int, optional
      The total number of items in all the lists combined. If -1 (default), the function calculates this automatically.
    - sort: bool
      If sort, keys are sorten, otherwise, the default order is used
    - chunk_size : int, optional
      The size of each chunk for processing the dictionary. Default is 2000.

    Returns:
    - keys : numpy.ndarray
      A NumPy array of integers representing the keys from the dictionary, repeated for each corresponding value.
    - vals : numpy.ndarray
      A NumPy array of integers representing the values from all lists in the dictionary, sorted by their keys.

    The function sorts the keys of the dictionary and processes the values list for each key in chunks,
    thereby potentially leveraging parallel execution capabilities for improved performance. Each value
    from the dictionary's lists is paired with its corresponding key in the 'keys' array, preserving the
    order dictated by the sorted keys.
    """
    if n_items < 0:
        n_items = sum([len(x) for x in typed_dict.values()]) #TODO: This can be done when computing the start indices
    keys = np.empty(n_items, dtype=np.int64)
    vals = np.empty(n_items, dtype=np.int64)

    keys_array = np.array(list(typed_dict.keys()), dtype=np.int64)
    if sort:
        keys_array = np.sort(keys_array)

    num_chunks = (len(keys_array) + chunk_size - 1) // chunk_size
    keys_array_chunks = np.array_split(keys_array, num_chunks)

    start_indices = [0]
    for i in range(1, len(keys_array_chunks)):
        start_indices.append(start_indices[i-1] + sum([len(typed_dict[k]) for k in keys_array_chunks[i-1]]))

    for chunk_i in numba.prange(num_chunks):
        chunkK = keys_array_chunks[chunk_i]
        start_idx = start_indices[chunk_i]
        j = 0
        for k in chunkK:
            for v in typed_dict[k]:
                keys[start_idx+j] = k
                vals[start_idx+j] = v
                j += 1
    raise NotImplementedError("I am getting some multithreading error!!!")
    return keys, vals

@njit(cache=True, nogil=True)
def typedList_to_numpy(l, dtype):
    a = np.empty(len(l), dtype=dtype)
    for i, v in enumerate(l):
        a[i] = v
    return a

def init_int_to_intList_dict():
    return Dict.empty(
        key_type=types.int64,
        value_type=types.ListType(types.int64)
    )

def init_int_to_int_dict():
    return Dict.empty(
        key_type=types.int64,
        value_type=types.int64
    )


def _test0():
    # Example usage
    typed_dict = Dict.empty(
        key_type=types.int64,
        value_type=types.ListType(types.int64)
    )


    # Add values to keys
    typed_dict_add_keyList_valList(typed_dict, List([1, 2, 3]), List([10, 20, 30]))
    print("After adding values:", typed_dict)

    # Read values from keys
    values = typed_dict_read_values_from_keysList(typed_dict, List([1, 2, 3]))
    print("Read values:", values)


    # Add values to keys
    typed_dict_add_keyList_valList(typed_dict, np.array([1, 2, 2, 3]), np.array([10, 20, 20, 30]))
    print("After adding values:", typed_dict)

    # Read values from keys
    values = typed_dict_read_values_from_keysList(typed_dict, np.array([1, 2, 2, 3]))
    print("Read values:", values)

    print(typed_dict_flatten_keys_vals_asarray(typed_dict))


@numba.njit(parallel=False)
def quaternion_distance(q1, q2):
    """
    Compute the distance between two rotations represented by unit quaternions.

    Parameters:
    - q1: A numpy array of shape (4,) representing the first quaternion.
    - q2: A numpy array of shape (4,) representing the second quaternion.

    Returns:
    - d(q1,q2)=1−⟨q1,q2⟩**2 which is equal to (1−cosθ)/2 and serves as a distance
    """
    # Normalize the quaternions to ensure they are unit quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dist = 1 - np.dot(q1, q2) ** 2

    return dist

def warmup():
    d = init_int_to_intList_dict()
    n = int(1e2)
    keys = np.random.randint(0, n//10, size=(n,)) #np.arange(n)
    vals = np.random.randint(0, 10, size=(n,)) #np.arange(n)
    typed_dict_add_keyList_valList(d, keys, vals)
    _a = typed_dict_flatten_keys_vals_asarray(d)
    # _b = parallel_typed_dict_flatten_keys_vals_asarray(d)
    typedList_to_numpy(d, keys.dtype)

    typed_dict_read_values_from_keysList(d, keys)
warmup()

def _test1():

    import time
    print("Filling dictionary")
    d = init_int_to_intList_dict()
    n = int(1e8)
    keys = np.random.randint(0, n//10, size=(n,)) #np.arange(n)
    vals = np.random.randint(0, int(2e5), size=(n,)) #np.arange(n)
    typed_dict_add_keyList_valList(d, keys, vals)
    print("Sequential code launched!", flush=True)
    t0 = time.time()
    _a = typed_dict_flatten_keys_vals_asarray(d)
    print(time.time()-t0, flush=True)
    print("Parallel code launched!", flush=True)
    t0 = time.time()
    _b = parallel_typed_dict_flatten_keys_vals_asarray(d)
    print(time.time() - t0, flush=True)

    assert np.isclose(_a, _b).all()

if __name__ == "__main__":

    # _test0()
    _test1()