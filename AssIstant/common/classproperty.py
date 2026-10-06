class classproperty:
    def __init__(self, fget=None):
        self.name = getattr(fget, "__name__", None)
        self._getter = fget

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if self._getter is None:
            raise AttributeError(f"ungettable attribute: \'{self.name}\'")

        if instance is not None and isinstance(instance, type):
            owner = instance
        elif owner is None:
            owner = type(instance)

        return self._getter(owner)


if __name__ == "__main__":
    class MyClass:
        _name = None

        @classmethod
        @classproperty
        def name(cls):
            return cls._name

    class YouClass(MyClass):
        _name = "YouClass"

    # MyClass.name = "MyClass"
    print(MyClass.name)
    print(YouClass.name)
