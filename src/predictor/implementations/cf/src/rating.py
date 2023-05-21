# author: Arnoud De Jonge

class Rating:
    def __init__(self, user_id: int, business_id: int, score: float):
        self.user_id = user_id
        self.business_id = business_id
        self.score = score

    def __str__(self):
        return f'Rating: {self.user_id}-{self.business_id}-{self.score}\n'
