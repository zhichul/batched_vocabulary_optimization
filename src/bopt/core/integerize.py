#!/usr/bin/env python3

from typing import (Dict, Generic, Iterable, Iterator, List, Optional, TypeVar, overload, Union)

T = TypeVar("T")  # see https://mypy.readthedocs.io/en/stable/generics.html


class Integerizer(Generic[T]):
    """
    A collection of distinct object types, such as a vocabulary or a set of parameter names,
    that are associated with consecutive ints starting at 0.

    Example usage:

    >>> from integerize import Integerizer
    >>> vocab: Integerizer[str]                       # type hint saying that the objects will be strings
    >>> vocab = Integerizer(['','hello','goodbye'])   # lets the empty string '' be 0
    >>> vocab.index('goodbye')                        # convert from word to int
    2
    >>> vocab[2]                                      # convert from int back to word
    'goodbye'
    >>> sentence = ('hello','world','if','world','you','be')
    >>> [vocab.index(w) for w in sentence]    # convert from words to ints (or None if OOV)
    [1, None, None, None, None, None]
    >>> [vocab.index(w, add=True) for w in sentence]  # expand vocabulary on demand (so no OOVs)
    [1, 3, 4, 3, 5, 6]
    >>> [vocab[i] for i in [1, 3, 4, 3, 5, 6]]        # convert from ints back to words
    ['hello', 'world', 'if', 'world', 'you', 'be']
    >>> len(vocab)                                    # vocab size (not including OOV)
    7
    >>> vocab[:]    # show all 7 word types, in order of their ints
    ['', 'hello', 'goodbye', 'world', 'if', 'you', 'be']
    >>> 'world' in vocab, 'mars' in vocab
    (True, False)
    """

    # If you are unfamiliar with the special __ method names, check out
    # https://docs.python.org/3/reference/datamodel.html#special-method-names .

    def __init__(self, iterable: List[T] = [], unk_token="[UNK]"):
        """
        Initialize the collection to the empty set, or to the set of *unique* objects in its argument
        (in order of first occurrence).
        """
        # Set up a pair of data structures to convert objects to ints and back again.
        self._objects: List[T] = []  # list of all unique objects that have been added so far
        self._indices: Dict[T, int] = {}  # maps each object to its integer position in the list
        # Add any objects that were given.
        self.update(iterable)

        # REMARK: Because of the way that dictionaries in Python are
        # implemented, this class actually stores each object twice
        # (both copies being represented by the same pointer) and
        # stores each integer twice.  That's wasteful!  The efficient
        # method would be to use a set of all the objects, which
        # internally contains an efficient integerizer that stores
        # each object and each integer only once.  Unfortunately,
        # Python's built-in set API doesn't give us access to the
        # integer indices that the set uses internally.
        self.unk_token = unk_token

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Integerizer):
            return self._objects == other._objects  # other._objects is List[Unknown] but that is ok since `==` allows any object
        else:
            return False

    def __hash__(self):
        return hash(tuple(self._objects))

    def __len__(self) -> int:
        """
        Number of objects in the collection.
        """
        return len(self._objects)

    def __iter__(self) -> Iterator[T]:
        """
        Iterate over all the objects in the collection.
        """
        return iter(self._objects)

    def __contains__(self, obj: T) -> bool:
        """
        Does the collection contain this object?  (Implements `in`.)
        """
        return self.index(obj) is not None

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> List[T]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, List[T]]:
        """
        Return the object with a given index.
        (Implements subscripting, e.g., `my_integerizer[3]` and `my_integerizer[3:5]`.)
        """
        return self._objects[index]

    def index(self, obj: T, add: bool = False, unk=False) -> Optional[int]:
        """
        The integer associated with a given object, or `None` if the object is not in the collection (OOV).
        Use `add=True` to add the object if it is not present.
        """
        try:
            return self._indices[obj]
        except KeyError:
            if not add:
                if not unk:
                    return None
                else:
                    print(f"WARNING: {self.unk_token} used for {obj}")
                    return self.index(self.unk_token)

            # add the object to both data structures
            i = len(self)
            self._objects.append(obj)
            self._indices[obj] = i
            return i

    def add(self, obj: T) -> None:
        """
        Add the object if it is not already in the collection.
        Similar to `set.add` (or `list.append`).
        """
        self.index(obj, add=True)  # call for side effect; ignore return value

    def update(self, iterable: Iterable[T]) -> None:
        """
        Add all the objects if they are not already in the collection.
        Similar to `set.update` (or `list.extend`).
        """
        for obj in iterable:
            self.add(obj)

    def __repr__(self) -> str:
        if len(self) < 42:
            return f"Integerizer({', '.join([f'{obj}:{self.index(obj)}' for obj in self._objects])})"
        else:
            return f"Integerizer({len(self)})"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
