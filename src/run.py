#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Файл для запуска FastAPI сервера из командной строки с настройками

import os
import argparse
import uvicorn
import dotenv

# Загрузка переменных окружения
dotenv.load_dotenv()

# Аргументы командной строки
parser = argparse.ArgumentParser(description='ANPR FastAPI Server')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
parser.add_argument('--port', type=int, default=8000, help='Port to bind')
parser.add_argument('--reload', action='store_true', help='Enable auto-reload on file changes')
parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
parser.add_argument('--log-level', type=str, default='info', help='Log level (debug, info, warning, error, critical)')
parser.add_argument('--database-url', type=str, help='Database URL (e.g. sqlite:///./anpr_data.db)')
parser.add_argument('--claude-api-key', type=str, help='Claude API key')
parser.add_argument('--claude-model', type=str, help='Claude model to use')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')

# Парсим аргументы командной строки
args = parser.parse_args()

# Устанавливаем переменные окружения из аргументов командной строки
if args.database_url:
    os.environ['DATABASE_URL'] = args.database_url
if args.claude_api_key:
    os.environ['CLAUDE_API_KEY'] = args.claude_api_key
if args.claude_model:
    os.environ['CLAUDE_MODEL'] = args.claude_model
if args.debug:
    os.environ['DEBUG'] = 'true'

if __name__ == "__main__":
    print(f"Запуск ANPR сервера на {args.host}:{args.port}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level
    )
