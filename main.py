from aiogram.utils import executor
from aiogram.types import BotCommand
import os

from create_bot import ans
from Client.client_router import register_client_router
from Client.client_router import PATH_WORKSPACE


def create_workspace():
    try:
        os.mkdir(PATH_WORKSPACE)
    except:
        pass


if ans:
    bot = ans[0]
    dp = ans[1]
    storage = ans[2]


async def on_startup(_):
    bot_commands = [
        BotCommand(command="/help", description="help"),
    ]
    await bot.set_my_commands(bot_commands)
    create_workspace()
    await bot.set_webhook(url='', allowed_updates=["message", "inline_query", "callback_query"])
    print('Бот запущен')


async def on_shutdown(_):
    await storage.close()
    await (await bot.get_session()).close()


if __name__ == '__main__':
    register_client_router(dp)
    executor.start_polling(dp, on_startup=on_startup, on_shutdown=on_shutdown)