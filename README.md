ANPR FastAPI Server - Система распознавания автомобильных номеров
Описание проекта
Этот проект представляет собой веб-сервер на базе FastAPI для системы распознавания автомобильных номеров (ANPR - Automatic Number Plate Recognition). Система позволяет загружать изображения автомобилей, автоматически обнаруживать и распознавать номерные знаки, сохранять результаты в базе данных и предоставлять доступ к ним через REST API.
Основные возможности

Загрузка и обработка изображений для распознавания автомобильных номеров
Автоматическое определение положения номерных знаков на изображении с использованием YOLO
Распознавание текста номерных знаков с использованием комбинации EasyOCR и Claude API
Сохранение всех результатов в базе данных
Возможность верификации и редактирования распознанных номеров
API для доступа к истории распознаваний и управления записями
Статистика по процессу распознавания
Поиск по базе номеров

Структура проекта
project/
├── server.py              # Основной FastAPI сервер
├── run.py                 # Скрипт для запуска сервера
├── src/
│   └── detector.py        # Модифицированный модуль распознавания
├── models/                # Модели для распознавания
│   ├── darknet-yolov3.cfg # Конфигурация YOLO
│   ├── lapi.weights       # Веса YOLO
│   └── classes.names      # Классы YOLO
├── uploads/               # Директория для загруженных изображений
├── detected_plates/       # Директория для вырезанных номеров
├── anpr_data.db           # SQLite база данных (по умолчанию)
├── .env                   # Переменные окружения
└── API_DOCUMENTATION.md   # Документация по API
Требования

Python 3.12+
OpenCV 4.8+
EasyOCR
FastAPI
SQLAlchemy
Uvicorn
Python-dotenv
Claude API ключ (опционально)

Установка и запуск
1. Клонирование репозитория
bashgit clone https://github.com/your-username/anpr-server.git
cd anpr-server
2. Создание виртуального окружения и установка зависимостей
bashpython -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Настройка переменных окружения
Создайте файл .env в корневой директории проекта:
DATABASE_URL=sqlite:///./anpr_data.db
CLAUDE_API_KEY=sk-ant-xxxxx
CLAUDE_MODEL=claude-3-sonnet-20240229
4. Запуск сервера
bashpython run.py --host 0.0.0.0 --port 8000 --reload
Сервер будет доступен по адресу http://localhost:8000
API
Полная документация по API доступна в файле API_DOCUMENTATION.md и через Swagger UI по адресу http://localhost:8000/docs или ReDoc http://localhost:8000/redoc после запуска сервера.
Основные эндпоинты

POST /api/upload - Загрузка и обработка изображения
GET /api/sessions - Получение списка сессий распознавания
GET /api/sessions/{session_id} - Получение информации о конкретной сессии
GET /api/plates - Получение списка распознанных номеров
GET /api/plates/{plate_id} - Получение информации о конкретном номере
PUT /api/plates/{plate_id} - Обновление информации о номере
DELETE /api/plates/{plate_id} - Удаление номера
GET /api/statistics - Получение статистики по распознаванию
GET /api/search - Поиск номеров

Работа с базой данных
По умолчанию сервер использует SQLite базу данных, расположенную в файле anpr_data.db. Для использования другой базы данных (например, PostgreSQL) измените значение DATABASE_URL в файле .env или передайте его через параметр командной строки --database-url.
Работа с Claude API
Для улучшения точности распознавания номеров система может использовать Claude API. Для этого необходимо получить API ключ у Anthropic и указать его в переменной среды CLAUDE_API_KEY или передать через параметр командной строки --claude-api-key.
Если ключ API не указан, система будет использовать только локальное распознавание через EasyOCR.
Дополнительные настройки
Дополнительные параметры можно указать при запуске сервера через аргументы командной строки:
bashpython run.py --help