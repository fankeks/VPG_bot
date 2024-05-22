from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

kb = [[KeyboardButton(text='/help')]]
client_kb = ReplyKeyboardMarkup(keyboard=kb,
                                resize_keyboard=True)