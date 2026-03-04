"""Min-heap (priority queue) — array-based binary heap."""


def _parent(i: int) -> int:
    return (i - 1) // 2


def _left(i: int) -> int:
    return 2 * i + 1


def _right(i: int) -> int:
    return 2 * i + 2


def _sift_up(h: list[int], i: int) -> None:
    while i > 0:
        p: int = _parent(i)
        if h[p] <= h[i]:
            break
        tmp: int = h[p]
        h[p] = h[i]
        h[i] = tmp
        i = p


def _sift_down(h: list[int], i: int, n: int) -> None:
    while True:
        smallest: int = i
        left: int = _left(i)
        right: int = _right(i)
        if left < n and h[left] < h[smallest]:
            smallest = left
        if right < n and h[right] < h[smallest]:
            smallest = right
        if smallest == i:
            break
        tmp: int = h[i]
        h[i] = h[smallest]
        h[smallest] = tmp
        i = smallest


def heap_push(h: list[int], val: int) -> None:
    """Add a value to the heap."""
    h.append(val)
    _sift_up(h, len(h) - 1)


def heap_pop(h: list[int]) -> int:
    """Remove and return the minimum value. Raises IndexError if empty."""
    n: int = len(h)
    if n == 0:
        raise IndexError("heap_pop from empty heap")
    result: int = h[0]
    n -= 1
    if n == 0:
        h.pop()
        return result
    h[0] = h[n]
    h.pop()
    _sift_down(h, 0, n)
    return result


def heap_peek(h: list[int]) -> int:
    """Return the minimum value without removing it. Raises IndexError if empty."""
    if len(h) == 0:
        raise IndexError("heap_peek at empty heap")
    return h[0]


def heap_size(h: list[int]) -> int:
    """Return the number of elements in the heap."""
    return len(h)


def heapify(data: list[int]) -> None:
    """Rearrange data in-place into a valid min-heap. O(n)."""
    n: int = len(data)
    i: int = _parent(n - 1)
    while i >= 0:
        _sift_down(data, i, n)
        i -= 1


def heap_sort(data: list[int]) -> list[int]:
    """Return a new sorted list using heapsort."""
    h: list[int] = list(data)
    heapify(h)
    out: list[int] = []
    while len(h) > 0:
        out.append(heap_pop(h))
    return out


def heap_push_pop(h: list[int], val: int) -> int:
    """Push val, then pop and return the minimum. More efficient than heap_push+heap_pop."""
    if len(h) > 0 and h[0] < val:
        result: int = h[0]
        h[0] = val
        _sift_down(h, 0, len(h))
        return result
    return val


def heap_replace(h: list[int], val: int) -> int:
    """Pop the minimum, then push val. Raises IndexError if empty."""
    if len(h) == 0:
        raise IndexError("heap_replace on empty heap")
    result: int = h[0]
    h[0] = val
    _sift_down(h, 0, len(h))
    return result
