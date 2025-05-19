# ANPR Server API Документация

## Основные эндпоинты

### 1. Загрузка изображения для распознавания

**URL**: `/api/upload`
**Метод**: `POST`
**Тип содержимого**: `multipart/form-data`

**Параметры запроса**:
- `file`: Изображение для распознавания номеров

**Пример запроса**:
```bash
curl -X POST \
  http://localhost:8000/api/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@car_image.jpg"
```

**Пример ответа**:
```json
{
  "session_id": 1,
  "filename": "car_image.jpg",
  "processing_time_ms": 542.3,
  "plates_detected": 2,
  "plates": [
    {
      "id": 1,
      "plate_number": "A123BC",
      "confidence": 0.87,
      "timestamp": "2025-05-19T12:34:56.789Z",
      "image_path": "detected_plates/plate_1621234567_0.jpg",
      "source_image_path": "uploads/1621234567_car_image.jpg",
      "position": {
        "x_min": 100,
        "y_min": 200,
        "x_max": 300,
        "y_max": 250
      },
      "processed_by": "local",
      "alternative_readings": "A123BC/A128BC",
      "is_verified": false,
      "is_deleted": false,
      "detection_id": 1
    },
    {
      "id": 2,
      "plate_number": "X789YZ",
      "confidence": 0.76,
      "timestamp": "2025-05-19T12:34:56.789Z",
      "image_path": "detected_plates/plate_1621234567_1.jpg",
      "source_image_path": "uploads/1621234567_car_image.jpg",
      "position": {
        "x_min": 400,
        "y_min": 300,
        "x_max": 600,
        "y_max": 350
      },
      "processed_by": "claude",
      "alternative_readings": null,
      "is_verified": false,
      "is_deleted": false,
      "detection_id": 1
    }
  ]
}
```

### 2. Получение списка сессий распознавания

**URL**: `/api/sessions`
**Метод**: `GET`

**Параметры запроса**:
- `skip`: Количество сессий для пропуска (пагинация)
- `limit`: Максимальное количество сессий для возврата

**Пример запроса**:
```bash
curl -X GET "http://localhost:8000/api/sessions?skip=0&limit=10"
```

**Пример ответа**:
```json
[
  {
    "id": 2,
    "filename": "car2.jpg",
    "timestamp": "2025-05-19T13:45:12.123Z",
    "processing_time": 634.5,
    "total_plates_detected": 1,
    "plates": [
      {
        "id": 3,
        "plate_number": "E456FG",
        "confidence": 0.91,
        "timestamp": "2025-05-19T13:45:12.123Z",
        "image_path": "detected_plates/plate_1621238712_0.jpg",
        "source_image_path": "uploads/1621238712_car2.jpg",
        "position": {
          "x_min": 150,
          "y_min": 220,
          "x_max": 350,
          "y_max": 270
        },
        "processed_by": "local",
        "alternative_readings": null,
        "is_verified": true,
        "is_deleted": false,
        "detection_id": 2
      }
    ]
  },
  {
    "id": 1,
    "filename": "car_image.jpg",
    "timestamp": "2025-05-19T12:34:56.789Z",
    "processing_time": 542.3,
    "total_plates_detected": 2,
    "plates": [
      {
        "id": 1,
        "plate_number": "A123BC",
        "confidence": 0.87,
        "timestamp": "2025-05-19T12:34:56.789Z",
        "image_path": "detected_plates/plate_1621234567_0.jpg",
        "source_image_path": "uploads/1621234567_car_image.jpg",
        "position": {
          "x_min": 100,
          "y_min": 200,
          "x_max": 300,
          "y_max": 250
        },
        "processed_by": "local",
        "alternative_readings": "A123BC/A128BC",
        "is_verified": false,
        "is_deleted": false,
        "detection_id": 1
      },
      {
        "id": 2,
        "plate_number": "X789YZ",
        "confidence": 0.76,
        "timestamp": "2025-05-19T12:34:56.789Z",
        "image_path": "detected_plates/plate_1621234567_1.jpg",
        "source_image_path": "uploads/1621234567_car_image.jpg",
        "position": {
          "x_min": 400,
          "y_min": 300,
          "x_max": 600,
          "y_max": 350
        },
        "processed_by": "claude",
        "alternative_readings": null,
        "is_verified": false,
        "is_deleted": false,
        "detection_id": 1
      }
    ]
  }
]
```

### 3. Получение информации о сессии по ID

**URL**: `/api/sessions/{session_id}`
**Метод**: `GET`

**Параметры пути**:
- `session_id`: ID сессии распознавания

**Пример запроса**:
```bash
curl -X GET "http://localhost:8000/api/sessions/1"
```

**Пример ответа**: (аналогично элементу из `/api/sessions`)

### 4. Получение списка распознанных номеров

**URL**: `/api/plates`
**Метод**: `GET`

**Параметры запроса**:
- `skip`: Количество номеров для пропуска (пагинация)
- `limit`: Максимальное количество номеров для возврата
- `plate_number`: Фильтр по номеру (опционально)
- `verified`: Фильтр по статусу верификации (опционально)

**Пример запроса**:
```bash
curl -X GET "http://localhost:8000/api/plates?skip=0&limit=20&plate_number=A123&verified=true"
```

**Пример ответа**:
```json
[
  {
    "id": 1,
    "plate_number": "A123BC",
    "confidence": 0.87,
    "timestamp": "2025-05-19T12:34:56.789Z",
    "image_path": "detected_plates/plate_1621234567_0.jpg",
    "source_image_path": "uploads/1621234567_car_image.jpg",
    "position": {
      "x_min": 100,
      "y_min": 200,
      "x_max": 300,
      "y_max": 250
    },
    "processed_by": "local",
    "alternative_readings": "A123BC/A128BC",
    "is_verified": true,
    "is_deleted": false,
    "detection_id": 1
  }
]
```

### 5. Получение информации о номере по ID

**URL**: `/api/plates/{plate_id}`
**Метод**: `GET`

**Параметры пути**:
- `plate_id`: ID распознанного номера

**Пример запроса**:
```bash
curl -X GET "http://localhost:8000/api/plates/1"
```

**Пример ответа**: (аналогично элементу из `/api/plates`)

### 6. Обновление информации о номере

**URL**: `/api/plates/{plate_id}`
**Метод**: `PUT`
**Тип содержимого**: `application/json`

**Параметры пути**:
- `plate_id`: ID распознанного номера

**Параметры запроса**:
```json
{
  "plate_number": "A123BC",
  "is_verified": true,
  "alternative_readings": "A123BC/A128BC"
}
```
(все поля опциональны)

**Пример запроса**:
```bash
curl -X PUT \
  "http://localhost:8000/api/plates/1" \
  -H "Content-Type: application/json" \
  -d '{"plate_number": "A123BC", "is_verified": true}'
```

**Пример ответа**: (обновленные данные номера)

### 7. Удаление номера

**URL**: `/api/plates/{plate_id}`
**Метод**: `DELETE`

**Параметры пути**:
- `plate_id`: ID распознанного номера

**Пример запроса**:
```bash
curl -X DELETE "http://localhost:8000/api/plates/1"
```

**Пример ответа**:
```json
{
  "message": "Номер успешно удален",
  "id": 1
}
```

### 8. Получение статистики

**URL**: `/api/statistics`
**Метод**: `GET`

**Пример запроса**:
```bash
curl -X GET "http://localhost:8000/api/statistics"
```

**Пример ответа**:
```json
{
  "total_plates": 100,
  "verified_plates": 75,
  "unrecognized_plates": 5,
  "recognition_rate": 0.95,
  "total_sessions": 42,
  "avg_confidence": 0.84,
  "recent_plates": [
    {
      "id": 100,
      "plate_number": "H789IJ",
      "confidence": 0.92,
      "timestamp": "2025-05-19T15:30:45.123Z",
      "image_path": "detected_plates/plate_1621245045_0.jpg",
      "source_image_path": "uploads/1621245045_car10.jpg",
      "position": {
        "x_min": 120,
        "y_min": 210,
        "x_max": 320,
        "y_max": 260
      },
      "processed_by": "local",
      "alternative_readings": null,
      "is_verified": false,
      "is_deleted": false,
      "detection_id": 42
    },
    /* еще 9 записей... */
  ]
}
```

### 9. Поиск номеров

**URL**: `/api/search`
**Метод**: `GET`

**Параметры запроса**:
- `query`: Поисковый запрос (номер или его часть)
- `limit`: Максимальное количество результатов

**Пример запроса**:
```bash
curl -X GET "http://localhost:8000/api/search?query=A123&limit=20"
```

**Пример ответа**:
```json
[
  {
    "id": 1,
    "plate_number": "A123BC",
    "confidence": 0.87,
    "timestamp": "2025-05-19T12:34:56.789Z",
    "image_path": "detected_plates/plate_1621234567_0.jpg",
    "source_image_path": "uploads/1621234567_car_image.jpg",
    "position": {
      "x_min": 100,
      "y_min": 200,
      "x_max": 300,
      "y_max": 250
    },
    "processed_by": "local",
    "alternative_readings": "A123BC/A128BC",
    "is_verified": true,
    "is_deleted": false,
    "detection_id": 1
  },
  {
    "id": 5,
    "plate_number": "A123DE",
    "confidence": 0.82,
    "timestamp": "2025-05-18T10:23:45.678Z",
    "image_path": "detected_plates/plate_1621134225_0.jpg",
    "source_image_path": "uploads/1621134225_car5.jpg",
    "position": {
      "x_min": 130,
      "y_min": 220,
      "x_max": 330,
      "y_max": 270
    },
    "processed_by": "local",
    "alternative_readings": null,
    "is_verified": false,
    "is_deleted": false,
    "detection_id": 5
  }
]
```

## Запуск сервера

Сервер можно запустить с помощью команды:

```bash
python run.py --host 0.0.0.0 --port 8000 --reload --log-level info --claude-api-key sk-ant-xxxx
```

### Параметры командной строки

- `--host`: Хост для привязки (по умолчанию: 0.0.0.0)
- `--port`: Порт для привязки (по умолчанию: 8000)
- `--reload`: Включить автоматическую перезагрузку при изменении файлов
- `--workers`: Количество рабочих процессов (по умолчанию: 1)
- `--log-level`: Уровень логирования (debug, info, warning, error, critical)
- `--database-url`: URL базы данных (например, sqlite:///./anpr_data.db)
- `--claude-api-key`: Ключ API Claude
- `--claude-model`: Модель Claude для использования
- `--debug`: Включить режим отладки

## Структура базы данных

### 1. Таблица `license_plates`

Хранит информацию о распознанных номерных знаках.

| Поле | Тип | Описание |
| --- | --- | --- |
| id | Integer | Первичный ключ |
| plate_number | String | Номер автомобиля |
| confidence | Float | Уверенность распознавания |
| timestamp | DateTime | Время распознавания |
| image_path | String | Путь к изображению номера |
| source_image_path | String | Путь к исходному изображению |
| x_min | Integer | Левая координата рамки |
| y_min | Integer | Верхняя координата рамки |
| x_max | Integer | Правая координата рамки |
| y_max | Integer | Нижняя координата рамки |
| processed_by | String | Метод обработки (local, claude, manual) |
| alternative_readings | String | Альтернативные варианты распознавания |
| is_verified | Boolean | Флаг верификации |
| is_deleted | Boolean | Флаг удаления |
| detection_id | Integer | Внешний ключ к сессии распознавания |

### 2. Таблица `detection_sessions`

Хранит информацию о сессиях распознавания.

| Поле | Тип | Описание |
| --- | --- | --- |
| id | Integer | Первичный ключ |
| filename | String | Имя исходного файла |
| timestamp | DateTime | Время сессии |
| processing_time | Float | Время обработки (мс) |
| total_plates_detected | Integer | Общее количество обнаруженных номеров |
