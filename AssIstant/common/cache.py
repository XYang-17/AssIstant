from typing import Any, Callable

def _cache_role(role: str) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        func._cache_role = role
        return func
    return decorator


class LRUCache:
    __none = object()
    __expired = object()

    # 根据传入参数计算cache_key并返回，默认组合所有输入参数
    make_key      = staticmethod(_cache_role('make_key'))
    # 根据输入参数提取get方法实际需要返回的cache内容，默认原样返回
    extract_value = staticmethod(_cache_role('extract_value'))
    # 根据输入参数构造set方法实际需要存入cache的内容，默认原样返回
    make_value    = staticmethod(_cache_role('make_value'))
    # 根据输入参数判断cache是否超时，默认永不超时
    is_expired    = staticmethod(_cache_role('is_expired'))

    def __init__(
        self,
        max_cache: int,
        eviction: int = 1
    ):
        if not isinstance(max_cache, int):
            raise TypeError(f"max_cache requires positive int, but {type(max_cache)} is provided")
        if max_cache <= 0:
            raise ValueError(f"max_cache requires positive int, but {max_cache} is provided")
        if not isinstance(eviction, int):
            raise TypeError(f"eviction_rate requires positive int, but {type(eviction)} is provided")
        if eviction <= 0:
            raise ValueError(f"eviction_rate requires positive int, but {eviction} is supported")

        self.__cache = dict()
        self.__max_cache = max_cache
        self.__eviction = max(1, min(max_cache // 2, eviction))

    @property
    def none(self) -> object:
        return self.__none

    @property
    def expired(self) -> object:
        return self.__expired

    @property
    def capacity(self) -> int:
        return self.__max_cache

    def cache_key(self, *args, **kwargs) -> Any:
        return self._make_key_impl(*args, **kwargs)

    def get(self, key) -> Any:
        if key not in self.__cache:
            return self.none
        value = self.__cache.pop(key)
        if self._is_expired_impl(value):
            return self.expired
        self.__cache[key] = value
        return self._extract_value_impl(value)

    def set(self, key, *args, **kwargs) -> Any:
        if key in self.__cache:
            self.__cache.pop(key)
        elif len(self.__cache) >= self.__max_cache:
            for cached_key in list(self.__cache)[:self.__eviction]:
                del self.__cache[cached_key]
        self.__cache[key] = self._make_value_impl(*args, **kwargs)

    def clear(self):
        self.__cache.clear()

    def __len__(self) -> int:
        return len(self.__cache)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, attr in list(vars(cls).items()):
            role = getattr(attr, '_cache_role', None)
            if role is not None:
                setattr(cls, f'_{role}_impl', attr)

    @staticmethod
    def _make_key_impl(key: Any) -> Any:
        return key

    @staticmethod
    def _extract_value_impl(cached_value: Any) -> Any:
        return cached_value

    @staticmethod
    def _make_value_impl(value: Any) -> Any:
        return value

    @staticmethod
    def _is_expired_impl(cached_value: Any) -> bool:
        return False


if __name__ == "__main__":
    from time import monotonic

    class MyCache(LRUCache):
        @LRUCache.make_key
        def _make_key(self, key: str) -> str:
            return key

        @LRUCache.extract_value
        def _extract_value(self, value) -> str:
            return value[0]

        @LRUCache.make_value
        def _make_value(self, value: str):
            return value, monotonic()

        @LRUCache.is_expired
        def _is_expired(self, value) -> bool:
            return monotonic() > value[1]

    cache = MyCache(10, 2)

    k = cache.cache_key("yangx")
    print(k)
    cache.set(k, "it's me")
    print(cache.get(k))
