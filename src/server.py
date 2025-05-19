#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# FastAPI сервер для системы распознавания автомобильных номеров (ANPR)

import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from detector import (
    init_ocr, recognize_license_plate, net, getOutputsNames, 
    confThreshold, nmsThreshold, inpWidth, inpHeight
)
import uvicorn
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("anpr-server")

# Настройка базы данных
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./anpr_data.db")
Base = declarative_base()

# Модели базы данных
class LicensePlate(Base):
    __tablename__ = "license_plates"
    
    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String)
    source_image_path = Column(String)
    x_min = Column(Integer)
    y_min = Column(Integer)
    x_max = Column(Integer)
    y_max = Column(Integer)
    processed_by = Column(String, default="local")  # local, claude, manual
    alternative_readings = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    
    detection_id = Column(Integer, ForeignKey("detection_sessions.id"))
    detection = relationship("DetectionSession", back_populates="plates")
    
    def to_dict(self):
        return {
            "id": self.id,
            "plate_number": self.plate_number,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "image_path": self.image_path,
            "source_image_path": self.source_image_path,
            "position": {
                "x_min": self.x_min,
                "y_min": self.y_min,
                "x_max": self.x_max,
                "y_max": self.y_max
            },
            "processed_by": self.processed_by,
            "alternative_readings": self.alternative_readings,
            "is_verified": self.is_verified,
            "is_deleted": self.is_deleted,
            "detection_id": self.detection_id
        }

class DetectionSession(Base):
    __tablename__ = "detection_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)  # в миллисекундах
    total_plates_detected = Column(Integer, default=0)
    
    plates = relationship("LicensePlate", back_populates="detection")
    
    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "total_plates_detected": self.total_plates_detected,
            "plates": [plate.to_dict() for plate in self.plates if not plate.is_deleted]
        }

# Создаем подключение к БД
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Создаем все таблицы в БД
Base.metadata.create_all(bind=engine)

# Pydantic модели для валидации данных
class PlatePosition(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    
    model_config = ConfigDict(
        from_attributes=True
    )

class LicensePlateCreate(BaseModel):
    plate_number: str
    confidence: float
    image_path: Optional[str] = None
    source_image_path: str
    position: PlatePosition
    processed_by: str = "local"
    alternative_readings: Optional[str] = None
    is_verified: bool = False
    
    model_config = ConfigDict(
        from_attributes=True
    )

class LicensePlateUpdate(BaseModel):
    plate_number: Optional[str] = None
    is_verified: Optional[bool] = None
    alternative_readings: Optional[str] = None
    
    model_config = ConfigDict(
        from_attributes=True
    )

class LicensePlateResponse(BaseModel):
    id: int
    plate_number: str
    confidence: float
    timestamp: datetime
    image_path: Optional[str]
    source_image_path: str
    position: PlatePosition
    processed_by: str
    alternative_readings: Optional[str]
    is_verified: bool
    is_deleted: bool
    detection_id: int
    
    model_config = ConfigDict(
        from_attributes=True
    )

class DetectionSessionCreate(BaseModel):
    filename: str
    processing_time: float
    total_plates_detected: int
    
    model_config = ConfigDict(
        from_attributes=True
    )

class DetectionSessionResponse(BaseModel):
    id: int
    filename: str
    timestamp: datetime
    processing_time: float
    total_plates_detected: int
    plates: List[LicensePlateResponse] = []
    
    model_config = ConfigDict(
        from_attributes=True
    )

# Зависимость для получения сессии БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Инициализация FastAPI
app = FastAPI(
    title="ANPR API",
    description="API для системы распознавания автомобильных номеров",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаем директории для сохранения изображений, если они не существуют
os.makedirs("uploads", exist_ok=True)
os.makedirs("detected_plates", exist_ok=True)

# Монтируем директории как статические файлы
app.mount("/images", StaticFiles(directory="uploads"), name="uploads")
app.mount("/plates", StaticFiles(directory="detected_plates"), name="detected_plates")

# Функция обработки изображения
async def process_image(image_data, filename, db: Session):
    # Конвертируем numpy array в изображение OpenCV
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")
    
    # Сохраняем оригинальное изображение
    # timestamp = int(time.time() * 1000)
    # source_path = f"uploads/{timestamp}_{filename}"
    # cv2.imwrite(source_path, frame)
    
    # Создаем сессию распознавания
    detection_session = DetectionSession(
        filename=filename,
        timestamp=datetime.utcnow(),
        processing_time=0,
        total_plates_detected=0
    )
    db.add(detection_session)
    db.commit()
    db.refresh(detection_session)
    
    # Засекаем время для измерения производительности
    start_time = time.time()
    
    # Создаем 4D blob из кадра
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    
    # Устанавливаем входные данные для нейронной сети
    net.setInput(blob)
    
    # Выполняем прямой проход для получения выходных данных
    outs = net.forward(getOutputsNames(net))
    
    # Обработка результатов
    plates_data = postprocess_detection(frame, outs, detection_session.id, db)
    
    # Измеряем время обработки
    processing_time = (time.time() - start_time) * 1000  # в миллисекундах
    
    # Обновляем сессию распознавания
    detection_session.processing_time = processing_time
    detection_session.total_plates_detected = len(plates_data)
    db.commit()
    
    # Возвращаем результаты
    return {
        "session_id": detection_session.id,
        "filename": filename,
        "processing_time_ms": processing_time,
        "plates_detected": len(plates_data),
        "plates": plates_data
    }

def postprocess_detection(frame, outs, detection_id, db: Session, source_path=""):
    """Обработка результатов детекции и распознавания номеров"""
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    classIds = []
    confidences = []
    boxes = []
    plates_data = []
    
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
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
        cv2.imwrite(plate_filename, plate_img)
        
        # Распознаем номер
        # Можно регулировать сохранение промежуточных результатов
        plate_number = recognize_license_plate(plate_img, save_debug=False)
        
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
        
        # Создаем запись в БД
        license_plate = LicensePlate(
            plate_number=plate_number,
            confidence=plate_confidence,
            timestamp=datetime.utcnow(),
            image_path=plate_filename,
            source_image_path=source_path,
            x_min=left,
            y_min=top,
            x_max=right,
            y_max=bottom,
            processed_by=processed_by,
            alternative_readings=alternative_readings,
            detection_id=detection_id
        )
        
        db.add(license_plate)
        db.commit()
        db.refresh(license_plate)
        
        # Добавляем информацию о распознанном номере
        plates_data.append(license_plate.to_dict())
    
    return plates_data

# Эндпоинты API
@app.post("/api/upload", response_model=Dict[str, Any])
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Загрузка и обработка изображения для распознавания номеров"""
    try:
        # Проверка на допустимый формат
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Допустимы только изображения")
        
        # Чтение файла
        image_data = await file.read()
        
        # Обработка изображения
        result = await process_image(image_data, file.filename, db)
        
        return result
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")

@app.get("/api/sessions", response_model=List[Dict[str, Any]])
async def get_sessions(
    skip: int = 0, 
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Получение списка сессий распознавания"""
    sessions = db.query(DetectionSession).order_by(DetectionSession.timestamp.desc()).offset(skip).limit(limit).all()
    return [session.to_dict() for session in sessions]

@app.get("/api/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Получение информации о сессии распознавания по ID"""
    session = db.query(DetectionSession).filter(DetectionSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return session.to_dict()

@app.get("/api/plates", response_model=List[Dict[str, Any]])
async def get_plates(
    skip: int = 0, 
    limit: int = 20,
    plate_number: Optional[str] = None,
    verified: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Получение списка распознанных номеров с возможностью фильтрации"""
    query = db.query(LicensePlate).filter(LicensePlate.is_deleted == False)
    
    if plate_number:
        query = query.filter(LicensePlate.plate_number.like(f"%{plate_number}%"))
    
    if verified is not None:
        query = query.filter(LicensePlate.is_verified == verified)
    
    plates = query.order_by(LicensePlate.timestamp.desc()).offset(skip).limit(limit).all()
    return [plate.to_dict() for plate in plates]

@app.get("/api/plates/{plate_id}", response_model=Dict[str, Any])
async def get_plate(
    plate_id: int,
    db: Session = Depends(get_db)
):
    """Получение информации о распознанном номере по ID"""
    plate = db.query(LicensePlate).filter(LicensePlate.id == plate_id, LicensePlate.is_deleted == False).first()
    if plate is None:
        raise HTTPException(status_code=404, detail="Номер не найден")
    return plate.to_dict()

@app.put("/api/plates/{plate_id}", response_model=Dict[str, Any])
async def update_plate(
    plate_id: int,
    plate_data: LicensePlateUpdate,
    db: Session = Depends(get_db)
):
    """Обновление информации о распознанном номере"""
    plate = db.query(LicensePlate).filter(LicensePlate.id == plate_id, LicensePlate.is_deleted == False).first()
    if plate is None:
        raise HTTPException(status_code=404, detail="Номер не найден")
    
    # Обновляем поля, которые были переданы
    if plate_data.plate_number is not None:
        plate.plate_number = plate_data.plate_number
    
    if plate_data.is_verified is not None:
        plate.is_verified = plate_data.is_verified
    
    if plate_data.alternative_readings is not None:
        plate.alternative_readings = plate_data.alternative_readings
    
    db.commit()
    db.refresh(plate)
    return plate.to_dict()

@app.delete("/api/plates/{plate_id}", response_model=Dict[str, Any])
async def delete_plate(
    plate_id: int,
    db: Session = Depends(get_db)
):
    """Удаление (мягкое) распознанного номера"""
    plate = db.query(LicensePlate).filter(LicensePlate.id == plate_id, LicensePlate.is_deleted == False).first()
    if plate is None:
        raise HTTPException(status_code=404, detail="Номер не найден")
    
    plate.is_deleted = True
    db.commit()
    
    return {"message": "Номер успешно удален", "id": plate_id}

@app.get("/api/statistics", response_model=Dict[str, Any])
async def get_statistics(
    db: Session = Depends(get_db)
):
    """Получение статистики по распознанным номерам"""
    # Общее количество распознанных номеров
    total_plates = db.query(func.count(LicensePlate.id)).filter(LicensePlate.is_deleted == False).scalar()
    
    # Количество проверенных номеров
    verified_plates = db.query(func.count(LicensePlate.id)).filter(
        LicensePlate.is_deleted == False,
        LicensePlate.is_verified == True
    ).scalar()
    
    # Количество нераспознанных номеров
    unrecognized_plates = db.query(func.count(LicensePlate.id)).filter(
        LicensePlate.is_deleted == False,
        LicensePlate.plate_number == "Не распознан"
    ).scalar()
    
    # Количество сессий
    total_sessions = db.query(func.count(DetectionSession.id)).scalar()
    
    # Средняя уверенность распознавания
    avg_confidence = db.query(func.avg(LicensePlate.confidence)).filter(
        LicensePlate.is_deleted == False,
        LicensePlate.plate_number != "Не распознан"
    ).scalar() or 0
    
    # Последние 10 распознанных номеров
    recent_plates = db.query(LicensePlate).filter(
        LicensePlate.is_deleted == False
    ).order_by(LicensePlate.timestamp.desc()).limit(10).all()
    
    return {
        "total_plates": total_plates,
        "verified_plates": verified_plates,
        "unrecognized_plates": unrecognized_plates,
        "recognition_rate": (total_plates - unrecognized_plates) / total_plates if total_plates > 0 else 0,
        "total_sessions": total_sessions,
        "avg_confidence": avg_confidence,
        "recent_plates": [plate.to_dict() for plate in recent_plates]
    }

@app.get("/api/search", response_model=List[Dict[str, Any]])
async def search_plates(
    query: str = Query(..., description="Поисковый запрос (номер или его часть)"),
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Поиск номеров по части номера"""
    plates = db.query(LicensePlate).filter(
        LicensePlate.is_deleted == False,
        LicensePlate.plate_number.like(f"%{query}%")
    ).order_by(LicensePlate.timestamp.desc()).limit(limit).all()
    
    return [plate.to_dict() for plate in plates]

@app.get("/", response_class=JSONResponse)
async def root():
    """Корневой эндпоинт API"""
    return {
        "status": "ok",
        "api_version": "1.0.0",
        "description": "API для системы распознавания автомобильных номеров"
    }

@app.on_event("startup")
async def startup_event():
    """Действия при запуске сервера"""
    # Инициализируем OCR
    init_ocr()
    logger.info("Сервер ANPR запущен и готов к работе")

if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
