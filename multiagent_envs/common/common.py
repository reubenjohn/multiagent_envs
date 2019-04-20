from typing import List, Tuple, Iterator, TypeVar

T = TypeVar('T')


def neighbours_generator(iterable: List[T]) -> Iterator[Tuple[T, T, T]]:
	prev_iter = iter([iterable[-1]] + iterable[:-1])
	curr_iter = iter(iterable)
	next_iter = iter(iterable[1:] + [iterable[0]])
	return zip(prev_iter, curr_iter, next_iter)
