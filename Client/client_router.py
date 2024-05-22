from aiogram import Dispatcher
from aiogram.types import Message
import os


from Client.client_keyboard.client_kb import client_kb


PATH_WORKSPACE = 'Workspace'
extensions = ['avi',
              'mp4']


async def cmd_start(message: Message):
    await message.answer('Отправьте видео\nВ одном сообщении - одно видео', reply_markup=client_kb)
    await message.delete()


async def save_photo(message: Message):
    try:
        if message.content_type == 'video':
            name = os.path.join(PATH_WORKSPACE, f'{message.video.file_unique_id}.mp4')
            await message.video.download(name)

        elif message.content_type == 'document':
            if not (message.document.file_name.split('.') in extensions):
                return
            name = os.path.join(PATH_WORKSPACE, f'{message.document.file_unique_id}.mp4')
            await message.document.download(name)
    except:
        return
    
    await message.answer_chat_action("typing")

    # TODO: Обработка видео

    # TODO: Отправка результата
    await message.answer(f'ЧСС: {77}', reply_markup=client_kb)

    # TODO: удаление временных файлов


async def delete_message(message: Message):
    await message.delete()


def register_client_router(dp: Dispatcher):
    dp.register_message_handler(cmd_start,
                                commands=['start'])

    dp.register_message_handler(cmd_start,
                                commands=['help'])

    dp.register_message_handler(save_photo,
                                content_types=['video', 'document'])

    dp.register_message_handler(delete_message)