import threading
import time
import os
import pyaudio
from src.constants.constants import AudioConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import snowboydecoder
except ImportError:
    snowboydecoder = None
    logger.error("未安装 snowboydecoder，请先安装 snowboy 依赖。")

class SnowboyWakeWordDetector:
    """基于 Snowboy 的唤醒词检测器"""
    def __init__(self, 
                 model_paths=None, 
                 sensitivity=0.5, 
                 sample_rate=AudioConfig.INPUT_SAMPLE_RATE, 
                 buffer_size=AudioConfig.INPUT_FRAME_SIZE):
        """
        model_paths: list[str]，每个唤醒词的 .pmdl/.umdl 路径
        sensitivity: float/list，灵敏度（0~1），可为单个或多个
        """
        self.model_paths = model_paths or []
        if isinstance(sensitivity, (float, int)):
            self.sensitivity = [str(sensitivity)] * len(self.model_paths)
        else:
            self.sensitivity = [str(s) for s in sensitivity]
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.on_detected_callbacks = []
        self.running = False
        self.paused = False
        self.detection_thread = None
        self.audio = None
        self.stream = None
        self.stream_lock = threading.Lock()
        self.detector = None
        self.enabled = bool(snowboydecoder and self.model_paths)

    def start(self, audio_codec_or_stream=None):
        if not self.enabled:
            logger.warning("Snowboy 唤醒词检测未启用或未配置模型")
            return False
        if self.running:
            logger.info("Snowboy 唤醒词检测已在运行")
            return True
        self.running = True
        self.paused = False
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        logger.info("Snowboy 唤醒词检测已启动")
        return True

    def stop(self):
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"关闭音频流失败: {e}")
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f"终止音频设备失败: {e}")
        self.stream = None
        self.audio = None
        logger.info("Snowboy 唤醒词检测已停止")

    def pause(self):
        if self.running and not self.paused:
            self.paused = True
            logger.info("Snowboy 检测已暂停")

    def resume(self):
        if self.running and self.paused:
            self.paused = False
            logger.info("Snowboy 检测已恢复")

    def is_running(self):
        return self.running and not self.paused

    def on_detected(self, callback):
        self.on_detected_callbacks.append(callback)

    def _trigger_callbacks(self, index):
        for cb in self.on_detected_callbacks:
            try:
                cb(self.model_paths[index], f"唤醒词索引: {index}")
            except Exception as e:
                logger.error(f"回调执行失败: {e}", exc_info=True)

    def _detection_loop(self):
        if not snowboydecoder:
            logger.error("未安装 snowboydecoder，无法检测唤醒词")
            return
        try:
            self.detector = snowboydecoder.HotwordDetector(
                self.model_paths, sensitivity=','.join(self.sensitivity), audio_gain=1)
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=AudioConfig.CHANNELS,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.buffer_size
            )
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                data = self.stream.read(self.buffer_size, exception_on_overflow=False)
                ans = self.detector.detector.RunDetection(data)
                if ans > 0:
                    logger.info(f"检测到唤醒词，索引: {ans-1}")
                    self._trigger_callbacks(ans-1)
                    self.detector.detector.Reset()
        except Exception as e:
            logger.error(f"Snowboy 检测循环异常: {e}", exc_info=True)
        finally:
            self.stop() 