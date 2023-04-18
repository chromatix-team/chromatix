def parse_learnable(self, learnable, **kwargs):
    def parse(key, value):
        if key in learnable:
            if callable(value):
                return self.param(f"_{key}", value)
            else:
                return self.param(f"_{key}", lambda _: value)
        else:
            return value

    return [parse(key, value) for key, value in kwargs.items()]
