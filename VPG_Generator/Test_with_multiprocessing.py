import cv2
import multiprocessing
import matplotlib.pyplot as plt

from VPG_Generator.VPGGenerator import VPGGenerator


def frame_handler(queue):
    """
    Метод параллельной обработки кадров
    :param queue: Очерь, в которую будут складываться кадры
    :return:
    """
    vpg_generator = VPGGenerator()
    vpg = []
    while True:
        # Если в очереди нет кадров продолжи ожидать
        if queue.empty():
            continue

        frame = queue.get()

        # Проверка на конец регистрации
        if frame is None:
            break

        value = vpg_generator.get_vpg_discret(frame)
        vpg.append(value)

    print(f'Длинна сигнала: {len(vpg)}')
    plt.plot(vpg)
    plt.show()


def test_with_multiprocessing():
    """
    Функция для тестирования реализации параллельной
    регистрации кадра и его преобразования в отсчёт ВПГ
    :param vpg_generator: Тестируемый объект
    :return: None
    """
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=frame_handler, args=(queue,))
    p.start()

    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()

        # Если файл закончился выйти из цикла
        if ret == False:
            cap.release()
            cv2.destroyAllWindows()
            break

        # Если нажата кнопка перестать показывать видео
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cap.release()
            cv2.destroyAllWindows()
            break

        cv2.imshow('Video', frame)
        queue.put(frame)

    queue.put(None)
    p.join()


if __name__ == '__main__':
    test_with_multiprocessing()