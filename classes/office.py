class Office(object):
    """"""

    def __init__(self, x: int, y: int, reward: float = 0) -> None:
        if not isinstance(x, int):
            raise ValueError('Coordinate x must be an integer.')

        if not isinstance(y, int):
            raise ValueError('Coordinate y must be an integer.')

        if x < 0:
            raise ValueError('Invalid value for coordinate x.')

        if y < 0:
            raise ValueError('Invalid value for coordinate y.')

        self.x = x
        self.y = y
        self.reward = reward

    def __str__(self) -> str:
        return f'Office located at ({self.x}, {self.y}) with reward of {self.reward} points.'
