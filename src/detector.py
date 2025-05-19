#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Модифицированный модуль detector.py с улучшенным API для интеграции с веб-сервером

import cv2 as cv
import numpy as np
import os
import time
import re
import logging
import dotenv
import json
import base64
import requests

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ANPR-detector")

# Загрузка переменных окружения
dotenv.load_dotenv()

# Константы для Claude API
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_API_TIMEOUT = 30  # секунды
CLAUDE_CONFIDENCE_THRESHOLD = 1.0  # Порог уверенности для обращения к Claude (всегда запрашиваем, если < 1.0)

# Проверка наличия поддержки GUI
has_gui_support = True
try:
    cv.namedWindow("Test", cv.WINDOW_NORMAL)
    cv.destroyWindow("Test")
except:
    has_gui_support = False
    logger.info("OpenCV собран без поддержки GUI, будет использован headless режим")

# Параметры детектора
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
ocrThreshold = 0.2   # OCR confidence threshold

inpWidth = 416  # Width of network's input image
inpHeight = 416 # Height of network's input image

# Получаем ключ API из переменных окружения
claude_api_key = os.environ.get("CLAUDE_API_KEY")
claude_model = os.environ.get("CLAUDE_MODEL", "claude-3-sonnet-20240229")

if claude_api_key:
    if not claude_api_key.startswith('sk-ant-'):
        logger.warning("Claude API ключ имеет неправильный формат. Ключ должен начинаться с 'sk-ant-'")
        logger.warning("Claude API не будет использоваться.")
        claude_api_key = None
else:
    logger.info("Claude API ключ не предоставлен, будет использоваться только локальное распознавание.")

# Пути к файлам моделей
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

# Пути к файлам детектора
classesFile = os.path.join(MODEL_DIR, "classes.names")
modelConfiguration = os.path.join(MODEL_DIR, "darknet-yolov3.cfg")
modelWeights = os.path.join(MODEL_DIR, "lapi.weights")

# Загрузка классов
classes = None
try:
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    logger.error(f"Файл классов не найден: {classesFile}")
    # Создаем базовый файл классов, если он отсутствует
    with open(classesFile, 'w') as f:
        f.write("license_plate\n")
    classes = ["license_plate"]

# Инициализация OCR
easyocr_reader = None

def init_ocr():
    """Инициализация EasyOCR при первом использовании"""
    global easyocr_reader
    try:
        import easyocr
        # Языки: английский и русский
        easyocr_reader = easyocr.Reader(['en', 'ru'], gpu=False)
        logger.info("EasyOCR инициализирован успешно")
        return True
    except ImportError:
        logger.error("EasyOCR не установлен. Установите: pip install easyocr torch")
        return False
    except Exception as e:
        logger.error(f"Ошибка инициализации EasyOCR: {e}")
        return False

# Загрузка модели YOLO
try:
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    # Проверка поддержки CUDA
    if cv.cuda.getCudaEnabledDeviceCount() > 0:
        logger.info("Используется CUDA для ускорения")
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
except Exception as e:
    logger.error(f"Ошибка при загрузке модели YOLO: {e}")
    # Создаем заглушки для файлов конфигурации, если они отсутствуют
    if not os.path.exists(modelConfiguration):
        with open(modelConfiguration, 'w') as f:
            f.write("# Placeholder for YOLOv3 configuration\n")
        logger.warning(f"Создан пустой файл конфигурации: {modelConfiguration}")
    
    # Перейти в запасной режим или завершить работу
    raise ValueError("Не удалось загрузить модель YOLO. Проверьте наличие файлов моделей.")

# Get the names of the output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    # Исправить обработку выходных слоев для совместимости с разными версиями OpenCV
    unconnected_layers = net.getUnconnectedOutLayers()
    
    # Проверяем формат данных - в новых версиях возвращается простой массив
    if isinstance(unconnected_layers[0], (list, tuple)):
        # Старая версия OpenCV (массив массивов)
        return [layersNames[i[0] - 1] for i in unconnected_layers]
    else:
        # Новая версия OpenCV (одномерный массив)
        return [layersNames[i - 1] for i in unconnected_layers]

def recognize_with_claude(plate_img):
    """Распознавание номера с помощью Claude Vision API"""
    global claude_api_key, claude_model
    
    if not claude_api_key:
        logger.warning("Claude API ключ не предоставлен. Используем только локальное распознавание.")
        return ""
    
    try:
        # Сохраняем изображение перед отправкой (для отладки)
        timestamp = int(time.time() * 1000)
        os.makedirs("detected_plates", exist_ok=True)
        debug_filename = f"detected_plates/claude_plate_{timestamp}.jpg"
        cv.imwrite(debug_filename, plate_img)
        
        # Преобразуем изображение в base64
        _, buffer = cv.imencode('.jpg', plate_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": claude_model,
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Определи номерной знак автомобиля на этом изображении. Выдай только текст номера, ничего больше."
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        }
        
        logger.info("Отправка запроса в Claude API...")
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=DEFAULT_API_TIMEOUT)
        
        # Подробный вывод ошибок
        if response.status_code != 200:
            logger.error(f"Claude API вернул статус {response.status_code}")
            error_details = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            logger.error(f"Детали ошибки: {error_details}")
            return ""
        
        result = response.json()
        
        # Извлечение текста из ответа Claude
        if "content" in result and len(result["content"]) > 0:
            for content_item in result["content"]:
                if content_item["type"] == "text":
                    text = content_item["text"].strip()
                    logger.info(f"Claude API распознал текст: {text}")
                    return text
        
        logger.warning("Claude API не вернул распознанный текст")
        return ""
    except Exception as e:
        logger.error(f"Ошибка Claude API: {e}")
        return ""

def preprocess_license_plate(plate_img):
    """Улучшенная предобработка для номерных знаков"""
    
    # Проверка размера изображения
    h, w = plate_img.shape[:2]
    if h < 10 or w < 20:  # Слишком маленькое изображение
        return None
    
    # УЛУЧШЕНИЕ: Более агрессивное масштабирование
    scale_factor = max(6, 120/w, 60/h)
    plate_img = cv.resize(plate_img, (int(w*scale_factor), int(h*scale_factor)))
    
    # Сохраняем оригинал после масштабирования
    processed = {'original': plate_img}
    
    # Преобразование в оттенки серого
    gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
    
    # Нормализация яркости и контрастности
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    processed['enhanced'] = enhanced
    
    # Удаление шума с сохранением краев
    denoised = cv.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    processed['denoised'] = denoised
    
    # Повышение резкости
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp = cv.filter2D(denoised, -1, kernel)
    processed['sharp'] = sharp
    
    # Создание бинарных версий
    _, binary_otsu = cv.threshold(sharp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    processed['binary_otsu'] = binary_otsu
    
    binary_adaptive = cv.adaptiveThreshold(sharp, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv.THRESH_BINARY, 11, 2)
    processed['binary_adaptive'] = binary_adaptive
    
    # НОВЫЕ МЕТОДЫ ПРЕДОБРАБОТКИ:
    
    # Высокий контраст
    high_contrast = cv.convertScaleAbs(plate_img, alpha=2.0, beta=0)
    processed['high_contrast'] = cv.cvtColor(high_contrast, cv.COLOR_BGR2GRAY)
    
    # Инвертированная бинаризация
    processed['inv_binary'] = cv.bitwise_not(binary_otsu)
    
    # Простая бинаризация с адаптивным порогом
    _, simple_thresh = cv.threshold(gray, int(np.mean(gray) * 0.7), 255, cv.THRESH_BINARY)
    processed['simple_thresh'] = simple_thresh
    
    # Обострение краев через морфологию
    kernel_morph = np.ones((2, 2), np.uint8)
    morph_gradient = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel_morph)
    processed['edges'] = morph_gradient
    
    return processed

# Очистка текста номера
def clean_license_text(text):
    """Очистка и нормализация текста номерного знака"""
    # Удаление пробелов и переносов строк
    text = text.strip().replace('\n', '').replace('\r', '')
    
    # Удаление спецсимволов
    text = re.sub(r'[^\w\d]', '', text)
    
    # Замена похожих символов
    replacements = {
        'O': '0', 'о': '0', 'О': '0',  # Буква О -> цифра 0
        'I': '1', 'l': '1', 'і': '1',  # Буква I -> цифра 1
        'Z': '2', 'z': '2',            # Буква Z -> цифра 2
        'B': '8', 'В': '8',            # Буква B -> цифра 8
        'D': '0', 'Ꭰ': '0',            # Буква D -> цифра 0
        'S': '5', 's': '5',            # Буква S -> цифра 5
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def recognize_license_plate(plate_img, save_debug=True):
    """Распознавание номерного знака с использованием EasyOCR и Claude API как резервного метода"""
    global easyocr_reader
    
    # Сохраняем оригинальное изображение для диагностики если включен режим сохранения
    if save_debug:
        timestamp = int(time.time() * 1000)
        os.makedirs("detected_plates", exist_ok=True)
        debug_filename = f"detected_plates/original_plate_{timestamp}.jpg"
        cv.imwrite(debug_filename, plate_img)
    
    # Проверяем инициализирован ли OCR
    if easyocr_reader is None:
        init_ocr()
        
        # Если OCR все еще не инициализирован, используем Claude API
        if easyocr_reader is None and claude_api_key:
            return recognize_with_claude(plate_img)
        elif easyocr_reader is None:
            logger.error("OCR не инициализирован и Claude API не доступен")
            return ""
    
    # Проверяем размер изображения
    if plate_img.shape[0] < 15 or plate_img.shape[1] < 30:
        return ""  # Слишком маленькое изображение
    
    # Обработка изображения
    processed_images = preprocess_license_plate(plate_img)
    if processed_images is None:
        return ""
    
    # Массив для хранения результатов распознавания
    results = []
    max_confidence = 0.0
    
    # Распознавание различных версий изображения
    for img_type, img in processed_images.items():
        # Сохранение версии для отладки
        if save_debug:
            timestamp = int(time.time() * 1000)
            filename = f"detected_plates/plate_{timestamp}_{img_type}.jpg"
            cv.imwrite(filename, img)
        
        try:
            # Попытка распознавания с разными параметрами
            try:
                # Стандартные параметры
                ocr_result = easyocr_reader.readtext(img)
            except:
                ocr_result = []
                
            # Если стандартные параметры не сработали, пробуем альтернативные
            if not ocr_result:
                try:
                    # Альтернативные параметры
                    ocr_result = easyocr_reader.readtext(
                        img, 
                        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        width_ths=0.5,
                        link_threshold=0.4,
                    )
                except:
                    ocr_result = []
            
            for detection in ocr_result:
                text = detection[1]  # Текст из EasyOCR
                confidence = detection[2]  # Уверенность
                
                # Обновляем максимальную уверенность
                max_confidence = max(max_confidence, confidence)
                
                # Очистка текста
                cleaned_text = clean_license_text(text)
                
                # Проверяем минимальные требования
                if cleaned_text and len(cleaned_text) >= 2 and confidence > ocrThreshold:
                    results.append((cleaned_text, confidence, img_type))
        except Exception as e:
            logger.error(f"Ошибка распознавания ({img_type}): {e}")
    
    # Обработка результатов
    if results:
        # Сортировка по уверенности
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Вывод результатов для отладки
        for result in results[:3]:
            logger.info(f"OCR результат: {result[0]} (уверенность: {result[1]:.2f}, метод: {result[2]})")
        
        # Если максимальная уверенность ниже порога Claude, пробуем API
        if max_confidence < CLAUDE_CONFIDENCE_THRESHOLD and claude_api_key:
            logger.info(f"Уверенность OCR ({max_confidence:.2f} < {CLAUDE_CONFIDENCE_THRESHOLD}), пробуем Claude API...")
            claude_result = recognize_with_claude(plate_img)
            if claude_result:
                logger.info(f"Claude API результат: {claude_result}")
                return claude_result
        
        # Возвращаем лучший результат
        best_result = results[0][0]
        
        # Улучшение коротких результатов
        if len(best_result) <= 3 and len(results) > 1:
            # Если первый результат слишком короткий, попробуем другие
            for result in results[1:]:
                if len(result[0]) > len(best_result):
                    best_result = result[0]
                    logger.info(f"Улучшенный результат: {best_result} (уверенность: {result[1]:.2f}, метод: {result[2]})")
                    break
        
        # Если есть альтернативные варианты, добавляем их в скобках
        alternatives = set(r[0] for r in results[1:3] if r[0] != best_result)
        if alternatives:
            best_result += f" ({'/'.join(alternatives)})"
        
        return best_result
    
    # Если OCR не справился, используем Claude API
    if claude_api_key:
        logger.info("OCR не смог распознать номер, пробуем Claude API...")
        claude_result = recognize_with_claude(plate_img)
        if claude_result:
            logger.info(f"Claude API результат: {claude_result}")
            return claude_result
    
    return ""

def detect_plates_in_image(image_path, save_debug=False):
    """Обнаружение и распознавание номеров на изображении"""
    try:
        # Загрузка изображения
        frame = cv.imread(image_path)
        if frame is None:
            logger.error(f"Не удалось загрузить изображение: {image_path}")
            return []
        
        # Создание 4D blob из кадра
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        
        # Устанавливаем вход в сеть
        net.setInput(blob)
        
        # Выполняем прямой проход, чтобы получить выходы выходных слоев
        outs = net.forward(getOutputsNames(net))
        
        # Результаты распознавания
        results = []
        
        # Обработка результатов
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        
        classIds = []
        confidences = []
        boxes = []
        
        # Сканируем все ограничивающие рамки из выходных данных сети
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        
        # Применяем non-maximum suppression для устранения перекрывающихся боксов
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        
        for i in indices:
            if isinstance(i, (list, tuple, np.ndarray)) and hasattr(i, '__len__') and len(i) > 0:
                i = i[0]
                
            box = boxes[i]
            left = max(0, box[0])
            top = max(0, box[1])
            width = box[2]
            height = box[3]
            right = min(frameWidth - 1, left + width)
            bottom = min(frameHeight - 1, top + height)
            
            # Извлекаем область номерного знака
            if bottom <= top or right <= left:
                continue
                
            plate_img = frame[top:bottom, left:right].copy()
            
            # Проверка минимальных размеров
            if plate_img.shape[0] < 15 or plate_img.shape[1] < 30:
                continue
            
            # Сохраняем изображение номера
            timestamp = int(time.time() * 1000)
            plate_filename = f"detected_plates/plate_{timestamp}_{i}.jpg"
            cv.imwrite(plate_filename, plate_img)
            
            # Распознаем номер
            plate_number = recognize_license_plate(plate_img, save_debug=save_debug)
            
            # Проверяем, получили ли мы номер
            if not plate_number:
                plate_number = "Не распознан"
                plate_confidence = 0.0
                processed_by = "local"
                alternative_readings = None
            else:
                # Проверяем, содержит ли результат альтернативные варианты
                if "(" in plate_number:
                    main_part, alternatives_part = plate_number.split("(", 1)
                    plate_number = main_part.strip()
                    alternative_readings = alternatives_part.rstrip(")").strip()
                else:
                    alternative_readings = None
                    
                plate_confidence = confidences[i]
                # Определяем, кто обработал номер
                if "Claude API" in plate_number:
                    processed_by = "claude"
                else:
                    processed_by = "local"
            
            # Добавляем результат
            results.append({
                "plate_number": plate_number,
                "confidence": float(confidences[i]),
                "position": {
                    "x_min": left,
                    "y_min": top,
                    "x_max": right,
                    "y_max": bottom
                },
                "image_path": plate_filename,
                "alternative_readings": alternative_readings,
                "processed_by": processed_by
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")
        return []

# API для интеграции
def process_image_for_api(image_data, save_intermediate=False):
    """Обработка изображения для API"""
    try:
        # Конвертируем байты в numpy array и затем в изображение OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Не удалось декодировать изображение")
            return []
        
        # Сохраняем входное изображение
        timestamp = int(time.time() * 1000)
        source_img_path = f"uploads/input_{timestamp}.jpg"
        os.makedirs("uploads", exist_ok=True)
        cv.imwrite(source_img_path, frame)
        
        # Создание 4D blob из кадра
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        
        # Устанавливаем вход в сеть
        net.setInput(blob)
        
        # Выполняем прямой проход, чтобы получить выходы выходных слоев
        outs = net.forward(getOutputsNames(net))
        
        # Результаты распознавания
        results = []
        
        # Обработка результатов
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        
        classIds = []
        confidences = []
        boxes = []
        
        # Сканируем все ограничивающие рамки из выходных данных сети
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        
        # Применяем non-maximum suppression для устранения перекрывающихся боксов
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        
        for i in indices:
            if isinstance(i, (list, tuple, np.ndarray)) and hasattr(i, '__len__') and len(i) > 0:
                i = i[0]
                
            box = boxes[i]
            left = max(0, box[0])
            top = max(0, box[1])
            width = box[2]
            height = box[3]
            right = min(frameWidth - 1, left + width)
            bottom = min(frameHeight - 1, top + height)
            
            # Извлекаем область номерного знака
            if bottom <= top or right <= left:
                continue
                
            plate_img = frame[top:bottom, left:right].copy()
            
            # Проверка минимальных размеров
            if plate_img.shape[0] < 15 or plate_img.shape[1] < 30:
                continue
            
            # Сохраняем изображение номера
            timestamp = int(time.time() * 1000)
            os.makedirs("detected_plates", exist_ok=True)
            plate_filename = f"detected_plates/plate_{timestamp}_{i}.jpg"
            cv.imwrite(plate_filename, plate_img)
            
            # Распознаем номер
            plate_number = recognize_license_plate(plate_img, save_debug=save_intermediate)
            
            # Проверяем, получили ли мы номер
            if not plate_number:
                plate_number = "Не распознан"
                plate_confidence = 0.0
                processed_by = "local"
                alternative_readings = None
            else:
                # Проверяем, содержит ли результат альтернативные варианты
                if "(" in plate_number:
                    main_part, alternatives_part = plate_number.split("(", 1)
                    plate_number = main_part.strip()
                    alternative_readings = alternatives_part.rstrip(")").strip()
                else:
                    alternative_readings = None
                    
                plate_confidence = confidences[i]
                # Определяем, кто обработал номер
                processed_by = "local"
                if "Claude API" in plate_number:
                    processed_by = "claude"
            
            # Добавляем результат
            results.append({
                "plate_number": plate_number,
                "confidence": float(confidences[i]),
                "bbox": [left, top, right, bottom],
                "position": {
                    "x_min": left,
                    "y_min": top,
                    "x_max": right,
                    "y_max": bottom
                },
                "image_path": plate_filename,
                "source_image_path": source_img_path,
                "alternative_readings": alternative_readings,
                "processed_by": processed_by
            })
        
        # Создаем отладочное изображение с выделенными номерами
        if save_intermediate:
            debug_img = frame.copy()
            for res in results:
                pos = res["position"]
                cv.rectangle(debug_img, 
                           (pos["x_min"], pos["y_min"]), 
                           (pos["x_max"], pos["y_max"]), 
                           (0, 255, 0), 2)
                # Добавляем текст на изображение
                cv.putText(debug_img, 
                         res["plate_number"], 
                         (pos["x_min"], pos["y_min"] - 10),
                         cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            debug_filename = f"uploads/debug_{timestamp}.jpg"
            cv.imwrite(debug_filename, debug_img)
        
        return results, source_img_path
    
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения для API: {e}")
        return [], None

# Инициализация при импорте (для проверки, что модель загружена)
if __name__ == "__main__":
    logger.info("Модуль детектора номеров успешно инициализирован")
    
    # Тест детектора
    if os.path.exists("test.jpg"):
        logger.info("Выполняем тест на изображении test.jpg")
        results = detect_plates_in_image("test.jpg", save_debug=True)
        logger.info(f"Найдено номеров: {len(results)}")
        for plate in results:
            logger.info(f"Номер: {plate['plate_number']}, уверенность: {plate['confidence']:.2f}")
