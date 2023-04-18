def parse_learnable(self, learnable, **kwargs):
    for key, value in kwargs.items():
        if key in learnable:
            assert callable(value), "when training this must be a callable"
            self.__setattr__(key, self.param(key, value))
        else:
            self.__setattr__(key, value)
