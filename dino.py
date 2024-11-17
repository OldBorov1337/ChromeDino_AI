from gymnasium.utils.env_checker import check_env
from mss import mss  # Захват экрана
import pydirectinput  # Команды управления
import cv2  # Обработка изображений
import numpy as np
import pytesseract  # OCR для распознавания текста "GAME OVER"
from matplotlib import pyplot as plt
import time  # Задержки
from gymnasium import Env  # Базовые компоненты среды (заменено на gymnasium)
from gymnasium.spaces import Box, Discrete

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1, 83, 100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {"top": 300, "left": 0, "width": 600, "height": 500}
        self.done_location = {"top": 250, "left": 400, "width": 1000, "height": 150}
        self.step_count = 0  # Счетчик шагов для контроля цикла

    def step(self, action):
        action_map = {
            0: "space",
            1: "down",
            2: "no_op"
        }
        if action != 2:
            pydirectinput.press(action_map[action])
            time.sleep(0.05)  # Добавлена небольшая пауза между действиями

        self.step_count += 1
        done, done_cap = self.get_done()  # Проверка завершения игры
        new_observation = self.get_observation()  # Новое наблюдение
        reward = 1  # Награда за каждый шаг, пока игра не закончилась

        # Определяем флаг `truncated`, если игра превышает лимит по шагам
        truncated = self.step_count >= 100
        if done or truncated:
            self.step_count = 0  # Сброс счетчика шагов при завершении эпизода

        info = {}
        return new_observation, reward, done, truncated, info

    def render(self):
        cv2.imshow("Game", np.array(self.cap.grab(self.game_location))[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()

    def close(self):
        cv2.destroyAllWindows()

    def reset(self, seed=None, options=None):
        # Инициализация seed для генератора случайных чисел
        super().reset(seed=seed)
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press("space")
        self.step_count = 0  # Сброс счетчика шагов
        return self.get_observation(), {}

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))
        channel = np.reshape(resized, (1, 83, 100))
        return channel

    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]
        done_strings = ["OVER", "GAME"]  # Добавили "GAME" в список завершений
        done = False
        res = pytesseract.image_to_string(done_cap)[:4].strip().upper()
        print(f"Распознанный текст: '{res}'")  # Отладка OCR результата
        if res in done_strings:
            done = True
        return done, done_cap


# Создание экземпляра среды
env = WebGame()

 # Проверка среды
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker

# Проверка среды
env_checker.check_env(env)

class  TrainAndLogginCallBack(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLogginCallBack, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
        return True

CHECKPOINT_DIR = './train'
LOG_DIR = './logs'

callback = TrainAndLogginCallBack(check_freq=1000, save_path=CHECKPOINT_DIR)

# Импорт алгоритма DQN
from stable_baselines3 import DQN
# Создаем модель DQN
model = DQN("CnnPolicy", env, tensorboard_log=LOG_DIR,
            verbose=1, buffer_size=1200000, learning_starts=1000)

# Запуск обучения
model.learn(total_timesteps=88000, callback=callback)
