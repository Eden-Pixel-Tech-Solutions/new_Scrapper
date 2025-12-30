@echo off
echo Saving worker logs to worker_debug.log...
docker-compose logs --no-color worker > worker_debug.log
echo Done! Check worker_debug.log
pause
