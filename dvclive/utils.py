def nested_set(d, keys, value):
    """Set d[keys[0]]...[keys[-1]] to `value`.

    Example:
    >>> d = {}
    >>> nested_set(d, ['person', 'address', 'city'], 'New York')
    >>> d
    {'person': {'address': {'city': 'New York'}}}

    From:
    https://stackoverflow.com/questions/13687924/setting-a-value-in-a-nested-python-dictionary-given-a-list-of-indices-and-value
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def nested_get(d, keys):
    """Get `value` in d[keys[0]]...[keys[-1]].

    Example:
    >>> d = {'person': {'address': {'city': 'New York'}}}
    >>> nested_get(d, ['person', 'address', 'city'])
    >>> d
    'New York'
    """
    value = d
    for key in keys:
        if key not in value:
            return None
        value = value[key]
    return value
