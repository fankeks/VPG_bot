from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage


class BotCreator:
    def __init__(self, path):
        self.__path = path

    def create(self):
        try:
            f = open(self.__path, 'r')
        except:
            return False

        TOKEN_API = f.read()
        f.close()

        if TOKEN_API is None:
            return False

        bot = Bot(TOKEN_API)
        storage = MemoryStorage()
        dp = Dispatcher(bot, storage=storage)
        return bot, dp, storage


creator = BotCreator('Token')
ans = creator.create()