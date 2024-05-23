import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

from VPG_Generator.VPGGenerator import VPGGenerator


PATH_DATA = 'Data'


def file_reader(file_path: str, visualize=True) -> tuple:
    """
    Функция прочтения видео
    :param file_path: Путь к файлу
    :param visualize: Проиграть видео или нет
    :return: (frames, fps) - Массив кадров и фпс
    """
    # Инициализируем необходимые переменные
    frames = []

    # Считываем файл
    cap = cv2.VideoCapture(file_path)
    print(int(cap.get(3)), int(cap.get(4)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    while (True):
        ret, frame = cap.read()

        # Если файл закончился выйти из цикла
        if ret == False:
            cap.release()
            if visualize:
                cv2.destroyAllWindows()
            break

        # Если нажата кнопка перестать показывать видео
        if visualize:
            cv2.imshow('Video', frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord(' '):
                visualize = False
        else:
            cv2.destroyAllWindows()

        #Добавляем frame
        frames.append(frame)

    return np.array(frames), float(fps)


def test_face_detector(vpg_generator, frames: list):
    """
    Метод тестирования выделения лица на изображении
    :param vpg_generator - Объект, который тестируем
    :param frames: - Список кадров
    :return: None
    """
    start = time.time()
    for frame in frames:
        frame, _ = vpg_generator.detect_face(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.destroyAllWindows()
            break
    print(f'Время test_face_detector: {time.time() - start}')


def test_get_landmarks(vpg_generator, frames: list):
    """
    Метод тестирования выделения контрольных точек на лице
    :param vpg_generator - Объект, который тестируем
    :param frames: - Список кадров
    :return: None
    """
    frame_width = 640
    frame_height = 480
    path = os.path.join("video.avi")
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    start = time.time()
    for frame in frames:
        face_frame, rectangle = vpg_generator.detect_face(frame)
        face_frame_gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        points = vpg_generator.get_landmarks(face_frame_gray, [])

        ver = [50, 33, 30, 29]
        hor = [4, 5, 6, 7, 8, 9, 10, 11]

        for i, point in enumerate(np.array(points)):
            point[0] += rectangle[0]
            point[1] += rectangle[1]
            frame = cv2.circle(frame, point,
                               radius=2,
                               color=(0, 255, 0),
                               thickness=-1)
            frame = cv2.putText(frame, str(i), point, cv2.FONT_HERSHEY_SIMPLEX ,
                                0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Video', frame)
        video.write(frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.destroyAllWindows()
            break
    video.release()
    print(f'Время test_get_landmarks: {time.time() - start}')


def test_get_segmented_frame(vpg_generator, frames: list):
    """
    Метод тестирования ...
    :param vpg_generator - Объект, который тестируем
    :param frames: - Список кадров
    :return: None
    """
    start = time.time()
    for frame in frames:
        vpg_frame = vpg_generator._get_segmented_frame(frame)
        vpg_frame = np.mean(vpg_frame[1]) / (np.mean(vpg_frame[0]) + np.mean(vpg_frame[2]))
        print(vpg_frame)
    print(f'Время test_get_segmented_frame: {time.time() - start}')


def test_get_report(vpg_generator, frames: list):
    """
    Метод тестирования генерации ВПГ
    :param vpg_generator - Объект, который тестируем
    :param frames: - Список кадров
    :return: vpg - Сигнал ВПГ
    """
    start = time.time()
    vpg = vpg_generator.get_report(frames)
    print(f'Время test_get_report: {time.time() - start}')
    plt.plot(vpg)
    plt.show()

    return vpg


def test(vpg_generator):
    for file_name in os.listdir('Data'):
        file_path = os.path.join(PATH_DATA, file_name)

        frames, fps = file_reader(file_path, visualize=False)
        print(f'fps: {fps}')
        print(f'Колличество кадров: {len(frames)}')
        print(f'Длительность: {len(frames) / fps}')
        print()
#########################################################################################################
        #Тестируем методы
        print('Тестирование:')
        #test_face_detector(vpg_generator, frames)
        test_get_landmarks(vpg_generator, frames)
        #test_get_segmented_frame(vpg_generator, frames)
        #vpg = test_get_report(vpg_generator, frames)

        #print(frames[0])
        #vpg_generator.get_vpg_discret(frames[54])

        #file_path = file_path.split('.')[0] + '.npy'
        #np.save(file_path, vpg)


if __name__ == '__main__':
    vpg_generator = VPGGenerator()
    test(vpg_generator)