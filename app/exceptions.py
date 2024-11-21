class TennisError(Exception):

    def __init__(self, message):
        super().__init__(message)
        self.__message = message
        
    def get_message(self):
        return self.__message